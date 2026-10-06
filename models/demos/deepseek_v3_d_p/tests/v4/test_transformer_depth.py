# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full-depth chunked-prefill accuracy for TtV4Transformer against the vLLM golden, real weights.

The golden's 56320-token prompt runs a chunk at a time through every layer of the model, free-running:
no layer is fed the golden's own input except where a segment starts. Each layer the trace kept
(0..9 and the last) is graded at its ``layer_floor``, and the last rank's output is taken end to end
through the checkpoint's LM head and compared with vLLM's next-token top-k, one prediction per chunk.

Flash runs in one process. Pro does not fit one Galaxy whole, and 10..60 leaves too little DRAM headroom
for a run that loads for hours first, so it runs in three: 0..9 from the embedding; 10..35 from the
golden's layer-9 output, writing layer 35's streams to disk; and 36..60 from that file. The file is an
exact (fp32) pipeline boundary, so 36..60 still sees only what the device computed from layer 9 on.
The two tail rows depend on each other and run in that order.

End-to-end metrics are logged, not asserted.
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
from models.demos.deepseek_v3_d_p.tests.v4.test_block import _pack_streams
from models.demos.deepseek_v3_d_p.tests.v4.test_transformer import (
    hidden_to_host,
    layer_floor,
    mesh_params,
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
_PRO_SPLIT = 36  # first layer of the second Pro tail process
_HANDOFF = Path(os.getenv("V4_DEPTH_HANDOFF_DIR", "generated/v4_depth_handoff")) / "pro"

_GALAXY = [p for p in mesh_params(DeepSeekV4ProConfig.FABRIC_PAYLOAD_SIZE) if p.id == "torus-xy-8x4"]


def _handoff_file(chunk: int) -> Path:
    return _HANDOFF / f"layer{_PRO_SPLIT - 1}_chunk{chunk:02d}.pt"


def _upload_streams(mesh_device, streams):
    """Host streams ``[1, S, n, D]`` -> the packed fp32 device streams a non-first rank takes."""
    ms = tuple(mesh_device.shape)
    return ttnn.from_torch(
        _pack_streams(streams, ms[1]),
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


def run_depth(mesh_device, device_params, num_links, model_config, first, count, floor, graded):
    """Run layers ``[first, first + count)`` over the whole golden prompt and grade ``graded`` layers.

    The first rank starts from the prompt's tokens. Otherwise the input is the golden's layer-9 output
    for layer 10, or the earlier tail process's handoff file for ``_PRO_SPLIT``. A rank that does not
    end the model writes its last layer's streams to the handoff dir when that layer feeds
    ``_PRO_SPLIT``. Asserts rather than skips: a skipped row reads as a green one.
    """
    gold = _GOLDEN[model_config]
    trace = golden.resolve_trace(gold)
    assert trace is not None, f"no golden trace at {gold.trace}; ${gold.trace_env} overrides the path"
    checkpoint = golden.resolve_checkpoint(gold)
    assert checkpoint is not None, f"no checkpoint at {gold.checkpoint}; ${' / $'.join(gold.ckpt_envs)} override it"

    last = first + count
    is_first, is_last = first == 0, last == model_config.NUM_LAYERS
    from_golden = not is_first and first - 1 in trace.kept_layers
    needed = graded + ([first - 1] if from_golden else [])
    missing = [i for i in needed if i not in trace.kept_layers]
    assert not missing, f"{trace.path.name} did not keep decoder_output for layers {missing}"
    if not is_first and not from_golden:
        assert first == _PRO_SPLIT, f"layer {first} has no golden input and no handoff"
        absent = [str(_handoff_file(c)) for c in range(N_CHUNKS) if not _handoff_file(c).is_file()]
        assert not absent, f"no handoff for layer {first}; run the preceding tail row first. Missing: {absent[:2]}"
    writes_handoff = not is_last and last == _PRO_SPLIT

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

    if writes_handoff:
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
        elif from_golden:
            streams = trace.decoder_output(first - 1, start, start + CHUNK).reshape(1, CHUNK, n, hidden)
            tt_in = _upload_streams(mesh_device, streams)
        else:
            tt_in = _upload_streams(mesh_device, torch.load(_handoff_file(chunk)))

        out = model(tt_in, actual_isl=CHUNK, actual_start=start, layer_tap=tap)
        ttnn.deallocate(tt_in)
        if is_last:
            final[chunk] = hidden_to_host(mesh_device, out)[0, CHUNK - 1]
        elif writes_handoff:
            torch.save(streams_to_host(mesh_device, out, n), _handoff_file(chunk))
        ttnn.deallocate(out)
        logger.info(f"  chunk {chunk} done (start={start}) in {time.monotonic() - t0:.0f} s")

    label = f"v4 depth {model_config.__name__} [{first}, {last})"
    failures = []
    for i in graded:
        truth = trace.decoder_output(i).reshape(1, SEQ_CACHE, n, hidden)
        _, pcc = comp_pcc(truth, per_layer.pop(i))
        del truth
        logger.info(f"[{label}] layer {i} PCC: {pcc:.6f}")
        if pcc < layer_floor(config, i, floor):
            failures.append(f"layer {i} {pcc:.6f} < {layer_floor(config, i, floor)}")
    if is_last:
        _end_to_end(trace, checkpoint, final, label)
    assert not failures, ", ".join(failures)


@pytest.mark.parametrize("mesh_device, device_params, num_links", _GALAXY, indirect=["mesh_device", "device_params"])
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_flash_transformer_depth(mesh_device, device_params, num_links):
    graded = list(range(10)) + [DeepSeekV4FlashConfig.NUM_LAYERS - 1]
    run_depth(
        mesh_device, device_params, num_links, DeepSeekV4FlashConfig, 0, DeepSeekV4FlashConfig.NUM_LAYERS, 0.99, graded
    )


@pytest.mark.parametrize("mesh_device, device_params, num_links", _GALAXY, indirect=["mesh_device", "device_params"])
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_pro_transformer_depth_head(mesh_device, device_params, num_links):
    run_depth(mesh_device, device_params, num_links, DeepSeekV4ProConfig, 0, 10, 0.98, list(range(10)))


@pytest.mark.parametrize("mesh_device, device_params, num_links", _GALAXY, indirect=["mesh_device", "device_params"])
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_pro_transformer_depth_tail_b1(mesh_device, device_params, num_links):
    run_depth(mesh_device, device_params, num_links, DeepSeekV4ProConfig, 10, _PRO_SPLIT - 10, 0.98, [])


@pytest.mark.parametrize("mesh_device, device_params, num_links", _GALAXY, indirect=["mesh_device", "device_params"])
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_pro_transformer_depth_tail_b2(mesh_device, device_params, num_links):
    last = DeepSeekV4ProConfig.NUM_LAYERS
    run_depth(
        mesh_device, device_params, num_links, DeepSeekV4ProConfig, _PRO_SPLIT, last - _PRO_SPLIT, 0.98, [last - 1]
    )
