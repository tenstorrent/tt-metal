# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full-depth chunked-prefill accuracy for TtV4Transformer against the vLLM golden, real weights.

The golden's 56320-token prompt runs a chunk at a time from the embedding, free-running: no layer is
fed the golden's own input. The trace keeps only layers 0..9 and the last, so those are the layers that
can be graded: 0..9 at ``layer_floor`` and the last at ``_LAST_LAYER_FLOOR``. The last rank's output is
also taken through the checkpoint's LM head and compared with vLLM's next-token top-k, one prediction
per chunk; that comparison is logged, not asserted.

Flash runs all 43 layers in one process and grades 0..9 and 42. Pro does not fit one Galaxy whole, so
it runs in three processes, each writing its last layer's streams to disk for the next.
``test_v4_pro_transformer_depth_head`` runs 0..9 from the embedding and grades each of them. The
end-to-end rows continue from its layer-9 file through 10..35 and 36..60 and grade only the last layer.
The files are exact (fp32) pipeline boundaries, so 60 sees only what the device computed from the
embedding on. The three rows depend on each other and run in that order.
"""

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.v4 import golden
from models.demos.deepseek_v3_d_p.tests.v4.helpers import (
    hidden_to_host,
    layer_floor,
    mesh_params,
    pack_streams,
    streams_to_host,
    upload_tokens,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v4 import TtV4Transformer
from models.demos.deepseek_v3_d_p.tt.v4.weights import (
    V4CheckpointLayers,
    load_checkpoint_tensors,
    v4_model_from_checkpoint,
)
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS

CHUNK = PREFILL_CHUNK_TOKENS
SEQ_CACHE = 55 * 1024  # 56320, the length the golden was captured at
N_CHUNKS = SEQ_CACHE // CHUNK
_GOLDEN = {DeepSeekV4ProConfig: golden.V4_PRO, DeepSeekV4FlashConfig: golden.V4_FLASH}
_HEAD = "head.weight"
_PRO_HEAD = 10  # layers 0..9, the ones the trace keeps before the last
_PRO_TAIL = ((_PRO_HEAD, 26), (36, 25))  # (first, count) of each process after the head
_HANDOFF = Path(os.getenv("V4_DEPTH_HANDOFF_DIR", "generated/v4_depth_handoff")) / "pro"
# The last layer is graded at its own floor: its PCC is carried by a few tokens whose norm is
# hundreds of times the median. It measures Flash L42 0.5919 to 0.6348 across hosts and bases. A model
# without an entry logs its last-layer PCC ungraded.
_LAST_LAYER_FLOOR = {DeepSeekV4FlashConfig: 0.62}


def _galaxy(model_config):
    return [p for p in mesh_params(model_config.FABRIC_PAYLOAD_SIZE) if p.id == "torus-xy-8x4"]


def _handoff_file(layer: int, chunk: int) -> Path:
    return _HANDOFF / f"layer{layer}_chunk{chunk:02d}.pt"


def _upload_streams(mesh_device, streams):
    """Host streams ``[1, S, n, D]`` -> the packed fp32 device streams a non-first rank takes."""
    ms = tuple(mesh_device.shape)
    return ttnn.from_torch(
        pack_streams(streams, ms[1]),
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=ms, dims=(2, 3)),
    )


def _end_to_end(trace, checkpoint, final, label):
    """Log the last rank's next-token predictions against vLLM's top-k at each chunk's last position.

    ``final`` is ``[N_CHUNKS, hidden]``, the device's final-norm output at those positions.
    """
    ref_idx = trace.topk("indices")
    ref_lp = trace.topk("logprobs")
    assert ref_idx.shape[0] == N_CHUNKS, f"{trace.path.name} has {ref_idx.shape[0]} top-k rows, want {N_CHUNKS}"
    assert trace.metadata["max_num_batched_tokens"] == CHUNK, "top-k rows are per vLLM step, which must be one chunk"
    assert int(ref_idx[-1, 0]) == trace.completion_token_ids[0], "trace's last top-1 is not its completion token"

    head = load_checkpoint_tensors(checkpoint, [_HEAD])[_HEAD].float()
    logits = final.float() @ head.T
    logp = torch.log_softmax(logits, dim=-1)
    k = ref_idx.shape[1]
    ours = logits.topk(k, dim=-1).indices

    top1 = ours[:, 0] == ref_idx[:, 0]
    overlap = torch.tensor([len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(ours, ref_idx)])
    lp_err = (logp.gather(1, ref_idx[:, :1]) - ref_lp[:, :1]).abs()[:, 0]
    for row in range(N_CHUNKS):
        logger.info(
            f"[{label}] step {row} pos {CHUNK * (row + 1) - 1}: top1 {int(ours[row, 0])} vs {int(ref_idx[row, 0])}, "
            f"top-{k} overlap {overlap[row]:.2f}, top1 logprob err {lp_err[row]:.4f}"
        )
    logger.info(
        f"[{label}] END-TO-END top-1 match {int(top1.sum())}/{N_CHUNKS}, mean top-{k} overlap {overlap.mean():.3f}, "
        f"top1 logprob err mean {lp_err.mean():.4f} max {lp_err.max():.4f}"
    )


def run_depth(mesh_device, device_params, num_links, model_config, first, count, floor, graded, write_handoff=False):
    """Run layers ``[first, first + count)`` over the whole golden prompt and grade ``graded`` layers.

    The first rank starts from the prompt's tokens, any other from the handoff file the process ending
    at layer ``first - 1`` wrote. ``write_handoff`` writes this process's last layer's streams for the
    next one. Asserts rather than skips: a skipped row reads as a green one.
    """
    gold = _GOLDEN[model_config]
    trace = golden.resolve_trace(gold)
    assert trace is not None, f"no golden trace at {gold.trace}; ${gold.trace_env} overrides the path"
    checkpoint = golden.resolve_checkpoint(gold)
    assert checkpoint is not None, f"no checkpoint at {gold.checkpoint}; ${' / $'.join(gold.ckpt_envs)} override it"

    last = first + count
    label = f"v4 depth {model_config.__name__} [{first}, {last})"
    is_first, is_last = first == 0, last == model_config.NUM_LAYERS
    missing = [i for i in graded if i not in trace.kept_layers]
    assert not missing, f"{trace.path.name} did not keep decoder_output for layers {missing}"
    if not is_first:
        absent = [
            str(_handoff_file(first - 1, c)) for c in range(N_CHUNKS) if not _handoff_file(first - 1, c).is_file()
        ]
        assert not absent, f"no handoff for layer {first}; run the preceding row first. Missing: {absent[:2]}"

    config = v4_hf_config(model_config, last, max_seq_len=SEQ_CACHE)
    n, hidden = config.hc_mult, config.hidden_size
    t0 = time.monotonic()
    state_dict = v4_model_from_checkpoint(config, checkpoint, load_embed=is_first, load_tail=is_last)
    state_dict["layers"] = V4CheckpointLayers(config, checkpoint, first, count)
    model = TtV4Transformer(
        mesh_device,
        config,
        model_config,
        state_dict,
        count,
        CHUNK,
        first_layer_idx=first,
        is_first_rank=is_first,
        is_last_rank=is_last,
        max_seq_len=SEQ_CACHE,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
    )
    del state_dict
    logger.info(f"built layers [{first}, {last}) in {(time.monotonic() - t0) / 60:.1f} min")

    if write_handoff:
        _HANDOFF.mkdir(parents=True, exist_ok=True)
    per_layer = {i: torch.zeros(1, SEQ_CACHE, n, hidden) for i in graded}
    final = torch.zeros(N_CHUNKS, hidden)
    for chunk in range(N_CHUNKS):
        t0 = time.monotonic()
        start = chunk * CHUNK

        def tap(i, h, start=start):
            if i in per_layer:
                per_layer[i][:, start : start + CHUNK] = streams_to_host(mesh_device, h, n)

        if is_first:
            tt_in = upload_tokens(mesh_device, trace.token_ids(CHUNK, start))
        else:
            tt_in = _upload_streams(mesh_device, torch.load(_handoff_file(first - 1, chunk)))

        out = model(tt_in, actual_isl=CHUNK, actual_start=start, layer_tap=tap)
        ttnn.deallocate(tt_in)
        if is_last:
            final[chunk] = hidden_to_host(mesh_device, out)[0, CHUNK - 1]
        elif write_handoff:
            torch.save(streams_to_host(mesh_device, out, n), _handoff_file(last - 1, chunk))
        ttnn.deallocate(out)
        logger.info(f"  chunk {chunk} done (start={start}) in {time.monotonic() - t0:.0f} s")

    failures = []
    for i in graded:
        truth = trace.decoder_output(i).reshape(1, SEQ_CACHE, n, hidden)
        dev = per_layer.pop(i)
        _, pcc = comp_pcc(truth, dev)
        logger.info(f"[{label}] layer {i} PCC: {pcc:.6f}")
        del truth, dev
        if i == model_config.NUM_LAYERS - 1:
            min_pcc = _LAST_LAYER_FLOOR.get(model_config)
        else:
            min_pcc = layer_floor(config, i, floor)
        if min_pcc is not None and pcc < min_pcc:
            failures.append(f"layer {i} {pcc:.6f} < {min_pcc}")
    if is_last:
        _end_to_end(trace, checkpoint, final, label)
    assert not failures, ", ".join(failures)


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _galaxy(DeepSeekV4FlashConfig), indirect=["mesh_device", "device_params"]
)
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_flash_transformer_depth(mesh_device, device_params, num_links):
    graded = list(range(10)) + [DeepSeekV4FlashConfig.NUM_LAYERS - 1]
    run_depth(
        mesh_device, device_params, num_links, DeepSeekV4FlashConfig, 0, DeepSeekV4FlashConfig.NUM_LAYERS, 0.99, graded
    )


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _galaxy(DeepSeekV4ProConfig), indirect=["mesh_device", "device_params"]
)
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_pro_transformer_depth_head(mesh_device, device_params, num_links):
    run_depth(
        mesh_device,
        device_params,
        num_links,
        DeepSeekV4ProConfig,
        0,
        _PRO_HEAD,
        0.98,
        list(range(_PRO_HEAD)),
        write_handoff=True,
    )


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _galaxy(DeepSeekV4ProConfig), indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("first, count", _PRO_TAIL, ids=[f"layers{f}to{f + c - 1}" for f, c in _PRO_TAIL])
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_pro_transformer_e2e(mesh_device, device_params, num_links, first, count):
    is_last = first + count == DeepSeekV4ProConfig.NUM_LAYERS
    graded = [DeepSeekV4ProConfig.NUM_LAYERS - 1] if is_last else []
    run_depth(
        mesh_device,
        device_params,
        num_links,
        DeepSeekV4ProConfig,
        first,
        count,
        0.98,
        graded,
        write_handoff=not is_last,
    )
