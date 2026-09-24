# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.2 MTP4 and MTP7 chunked prefill through the real ``TtPrefillTransformer``.

Drives the production device path and gates every level against a teacher-forced CPU reference.
:data:`SCHEDULE_AXIS` chooses how the request is cut up.
"""

from __future__ import annotations

import copy
import gc
import itertools
import os
import time
from pathlib import Path
from typing import Sequence

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.runners.runner_utils import MTP_PAD_TOKEN_ID, num_mtp_tokens
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import SparseMLAReference
from models.demos.deepseek_v3_d_p.reference.glm_5_2.mtp import glm_mtp_predictor_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank, num_full_indexer_layers
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows import MTPUnionEmbedding
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.tt_mtp import TtMTPPredictor
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import MTP_CACHE_ENV, MTP_CACHE_PREFIX, enable_mtp_indexer_slot
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor, prepare_prefill_mtp_tokens
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered
from tests.ttnn.utils_for_testing import assert_with_pcc

SP_AXIS, TP_AXIS = 0, 1


FUSED_MTP_PCC = 0.999

# The block-output allowance: a budget, not a measured floor.
MTP_MODULE_OUTPUT_PCC = 0.96

DISPATCH_BUFFER_CAPACITY_FACTOR = 8


CHUNK = 5 * 1024
NUM_CHUNKS = 3
TOTAL = CHUNK * NUM_CHUNKS

# A last chunk whose real end sits inside a tile.
PARTIAL_TAIL = 2540
# Two turns on one cache; the resume point is tile-aligned but not a chunk multiple.
MT_PREFIX = 3072
MT_TURN2 = 2500


def _one_turn(actual_isl: int, num_chunks: int = NUM_CHUNKS) -> list[tuple[int, int]]:
    """``num_chunks`` chunks of a single request of ``actual_isl`` tokens, starting at 0."""
    return [(i * CHUNK, actual_isl) for i in range(num_chunks)]


SCHEDULE_AXIS = {
    "provided-all": lambda k: _one_turn(TOTAL + k),
    "provided-half": lambda k: _one_turn(TOTAL + k // 2),
    "provided-none": lambda k: _one_turn(TOTAL),
    "partial": lambda k: _one_turn(2 * CHUNK + PARTIAL_TAIL),
    "multiturn": lambda k: [(0, MT_PREFIX), (MT_PREFIX, MT_PREFIX + MT_TURN2)],
}
"""Name -> K -> the ``(actual_start, actual_isl)`` of every chunk this test drives, in order.

``provided-*`` vary how much of the final chunk's lookahead is already in the stream; ``partial``
ends mid-chunk, and ``multiturn`` resumes one cache at a tile-aligned but not chunk-aligned start.
"""

MTP_LEVEL_AXIS = (4, 7)
"""The level counts this test runs: MTP4 and MTP7, the two shipping configurations.

MTP7 is free on the wire -- ``num_mtp_tokens`` rounds both up to the same tile -- so only the KVPE
depth, the levels replayed and the host cost differ. K = 1 is the debugging leg, in ``test_mtp.py``."""


for _k in MTP_LEVEL_AXIS:
    assert num_mtp_tokens(_k) >= _k, f"num_mtp_tokens({_k}) = {num_mtp_tokens(_k)} cannot cover {_k} levels"


def _shard_dims():
    dims = [None, None]
    dims[TP_AXIS] = -1
    dims[SP_AXIS] = -2
    return dims


def _from_device(t: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """``[1, 1, C/sp, H/tp]`` per chip -> ``[1, 1, C, H]`` in POSITION order.

    Valid because this test runs ``is_balanced=False``, where the input sharding is a plain reshape
    and concatenating the chips back along ``-2`` is its exact inverse.
    """
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=_shard_dims(), mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)


def mtp_chunk_stream(
    all_tokens: Sequence[int],
    chunk_start: int,
    chunk_size: int,
    n_mtp: int,
    num_levels: int,
    *,
    actual_isl: int | None = None,
) -> tuple[list[int], int, int]:
    """A whole prompt + a chunk's absolute start -> its ``(stream, real_len, provided_levels)``.

    The stream is ``chunk_size + n_mtp`` ids: this chunk's positions followed by the lookahead, with
    the pad sentinel at every global position at or past ``actual_isl``.
    """
    total = len(all_tokens)
    c, n, k = int(chunk_size), int(n_mtp), int(num_levels)
    s = int(chunk_start)
    isl = total if actual_isl is None else int(actual_isl)
    assert 0 <= s < isl, f"chunk starts at {s}, past the {isl} real tokens"
    assert n > 0, f"n_mtp must be positive, got {n}"
    assert isl <= total, f"actual_isl {isl} exceeds the {total} ids available"

    real = min(c, isl - s)
    stream = [MTP_PAD_TOKEN_ID if s + i >= isl else all_tokens[s + i] for i in range(c + n)]
    assert len(stream) == c + n, f"stream is {len(stream)}, expected {c} + {n}"
    provided = max(0, min(k, isl - (s + real)))
    return stream, real, provided


def assert_socket_rows(stream: Sequence[int], sp_factor: int, chunk_size: int, n_mtp: int) -> None:
    """Claim (1): each chip's row is the CONTIGUOUS slice of the stream at its own offset.

    Restated over the host lists: on device the trunk row and the lookahead row are separate blocks.
    """
    isl = chunk_size // sp_factor
    for c in range(sp_factor):
        trunk = list(stream[c * isl : (c + 1) * isl])
        mtp = list(stream[(c + 1) * isl : (c + 1) * isl + n_mtp])
        assert len(trunk) == isl and len(mtp) == n_mtp, (
            f"chip {c}: trunk {len(trunk)} + mtp {len(mtp)} ids, expected {isl} + {n_mtp}; the stream "
            f"is {len(stream)} long"
        )
        assert trunk + mtp == list(stream[c * isl : c * isl + isl + n_mtp]), (
            f"chip {c}'s trunk row and MTP row do not abut: joined they must be the contiguous "
            f"stream[{c * isl}:{c * isl + isl + n_mtp}]"
        )


def _mtp_union(transformer: TtPrefillTransformer, stream: Sequence[int], n_mtp: int, num_levels: int):
    """Build this chunk's :class:`MTPUnionEmbedding` the way the runtime does.

    Mirrors the runtime's first-rank branch: upload both id tensors, gather each with the model's own
    embedding, and hand the blocks to ``from_ids``. The union owns ``trunk`` and frees it.
    """
    chunk_ids = prepare_prefill_input_tensor(
        list(stream[: transformer.seq_len]),
        transformer.mesh_device,
        transformer.sp_factor,
        transformer.is_balanced,
        transformer.mesh_shape,
        transformer.sp_axis,
    )
    mtp_ids = prepare_prefill_mtp_tokens(
        list(stream),
        transformer.mesh_device,
        transformer.sp_factor,
        transformer.mesh_shape,
        transformer.sp_axis,
        num_mtp_tokens=n_mtp,
    )
    union = MTPUnionEmbedding.from_ids(chunk_ids, mtp_ids, transformer.mtp_embed_ids, num_levels=num_levels)
    ttnn.deallocate(chunk_ids)
    ttnn.deallocate(mtp_ids)
    return union


def _mtp_cache_dir(preferred: Path, fallback_root: Path) -> Path:
    """``preferred`` if this run can use it, else the same leaf under ``fallback_root``.

    ``TT_GLM52_MTP_TTNN_CACHE`` overrides it.
    """
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        if os.access(preferred, os.W_OK) or any(preferred.glob("*.tensorbin")):
            return preferred
        reason = "exists, holds no weights, and is not writable by this user"
    except PermissionError as exc:
        reason = f"cannot be created ({exc.strerror})"

    fallback = fallback_root / preferred.parent.name / preferred.name
    fallback.mkdir(parents=True, exist_ok=True)
    logger.warning(
        f"[mtp chunks] MTP weight cache {preferred} {reason}; falling back to {fallback}. "
        f"That directory starts empty, so this run BUILDS the MTP layer's cache (~5.5 GiB) instead "
        f"of loading it. Set {MTP_CACHE_ENV} to a writable root to choose the location yourself."
    )
    return fallback


def _next_token_fn(transformer: TtPrefillTransformer, actual_isl: int):
    """``H^k -> int``: the greedy token at the last real row, through the trunk's own LM head.

    Tells the TEST which id the device must have generated so the reference can embed it.
    """

    def next_token(h_normed):
        row = actual_isl - 1 if transformer.padding_side == "right" else transformer.seq_len - 1
        lm_head = transformer.lm_head
        raw, (device_id, token_offset) = lm_head(h_normed, row)
        logits_host = lm_head.logit_to_host(raw, device_id)
        assert (
            logits_host.shape[-1] == lm_head.vocab_size
        ), f"expected full vocab {lm_head.vocab_size}, got {logits_host.shape[-1]} -- TP concat may be broken"
        logits = lm_head.select_first_token(logits_host, token_offset)
        flat = logits.reshape(-1)
        assert (
            flat.numel() == transformer.lm_head.vocab_size
        ), f"expected full-vocab logits, got {flat.numel()} of {transformer.lm_head.vocab_size}"
        return int(torch.argmax(flat).item())

    return next_token


def _expected_window(full_seq: list[int], chunk_start: int, level: int, width: int) -> list[int]:
    """The one statement this test checks, written independently of the production code.

    Level ``k`` of the chunk at ``s`` sees ``full_seq[s + k + 1 : s + k + 1 + width]``, where ``width``
    is the chunk's real length -- one expression for interior, final and resumed chunks alike.
    """
    start = chunk_start + level + 1
    return list(full_seq[start : start + width])


def _host_window_embedding(embed_table: torch.Tensor, window_ids: list[int]) -> torch.Tensor:
    """The embedding the device MUST have produced for ``window_ids`` [C, H], derived here.

    Bit-identical, because the device's path from ids to window is arithmetic-free.
    """
    return embed_table[torch.tensor(window_ids, dtype=torch.long)]


_MESH_PARAMS = [
    pytest.param(
        (8, 4),
        torus_xy_device_params(
            fabric_payload_size=GLM52Config.FABRIC_PAYLOAD_SIZE,
            worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
        ),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    ),
]


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("num_layers", [1, 78], ids=["layers1", "layers78"])
@pytest.mark.parametrize("skip_pcc", [False, True], ids=["pcc", "nopcc"])
@pytest.mark.parametrize("mtp_levels", MTP_LEVEL_AXIS, ids=[f"mtp{k}" for k in MTP_LEVEL_AXIS])
@pytest.mark.parametrize("schedule", list(SCHEDULE_AXIS), ids=list(SCHEDULE_AXIS))
@pytest.mark.parametrize("variant", ["glm_5_2"], indirect=True, ids=["glm52"])
@pytest.mark.parametrize("use_pretrained", [True], ids=["pretrained"], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_mtp_transformer_chunks(
    variant,
    config_only,
    model_path,
    weight_cache_path,
    mesh_device,
    device_params,
    num_links,
    num_layers,
    mtp_levels,
    schedule,
    use_pretrained,
    mtp_cfg,
    mtp_state_dict,
    mtp_layer_state_dict,
    skip_pcc,
    monkeypatch,
):
    """GLM-5.2 MTP4/MTP7 chunked prefill end to end: every window, every level, exact ids.

    Four claims, most-local first: the stream the socket delivers, every level's output against the
    teacher-forced reference, row 0 of every window, and the last chunk's generated tokens.
    """
    torch.manual_seed(42)
    if not skip_pcc and num_layers == 78 and schedule == "provided-all":
        pytest.skip("pcc-layers78-provided-all adds no path the interior chunks and layers1 do not")
    NUM_LEVELS = mtp_levels
    N_MTP = num_mtp_tokens(NUM_LEVELS)
    CHUNKS = SCHEDULE_AXIS[schedule](NUM_LEVELS)
    for _i, (_s, _isl) in enumerate(CHUNKS):
        assert _s % ttnn.TILE_SIZE == 0, f"chunk {_i} starts at {_s}, which is not tile-aligned"
        assert _s < _isl <= TOTAL + NUM_LEVELS, f"chunk {_i} at {_s} declares actual_isl {_isl}"
    for (_ps, _pisl), (_s, _isl) in zip(CHUNKS, CHUNKS[1:]):
        assert _s in (_ps + CHUNK, _pisl), (
            f"chunk at {_s} neither continues the chunk at {_ps} nor resumes the previous turn at " f"its end {_pisl}"
        )
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")

    topology = per_axis_topology(device_params["fabric_config"])
    mesh_shape = list(mesh_device.shape)
    sp_factor, tp_factor = mesh_shape[SP_AXIS], mesh_shape[TP_AXIS]

    config = copy.copy(config_only)
    config.max_seq_len = TOTAL
    full_depth = int(config_only.num_hidden_layers)
    assert 1 <= num_layers <= full_depth, f"trunk depth {num_layers} outside [1, {full_depth}]"
    shallow = num_layers < full_depth

    if shallow:
        config.indexer_types = list(config_only.indexer_types)[:num_layers]

    mtp_block_idx = max(num_layers, variant.model_config.NUM_DENSE_LAYERS)
    layer_idx = enable_mtp_indexer_slot(config, mtp_block_idx)
    assert layer_idx == mtp_block_idx >= num_layers, (
        f"the MTP block must sit past the trunk's layers [0, {num_layers}) so it cannot claim one of "
        f"their cache keys; got {layer_idx}"
    )
    assert layer_idx >= variant.model_config.NUM_DENSE_LAYERS, (
        f"TtPrefillBlock derives is_moe = layer_idx >= {variant.model_config.NUM_DENSE_LAYERS}, so an "
        f"MTP block at {layer_idx} would be built DENSE while its weights (checkpoint layer "
        f"{mtp_cfg.mtp_layer_idx}) are a 256-expert MoE"
    )
    if not shallow:
        assert (
            layer_idx == mtp_cfg.mtp_layer_idx == num_layers
        ), f"at full depth the MTP block sits on the checkpoint's own layer {mtp_cfg.mtp_layer_idx}, got {layer_idx}"
    trunk_stride = full_indexer_rank(config, num_layers + NUM_LEVELS)
    mtp_stride = num_full_indexer_layers(config)
    assert trunk_stride == mtp_stride, (
        f"index-K cache stride disagrees: trunk {trunk_stride} vs MTP {mtp_stride}. They share one "
        "cache, so update_padded_kv_cache raises TT_FATAL 'cache batch dim must be a multiple of "
        "num_layers' on the first indexer write -- before any assertion below is reached."
    )
    hidden = config.hidden_size
    assert hidden == mtp_cfg.hidden_size

    logger.info(
        f"[mtp chunks] mesh={mesh_shape} schedule={schedule}{CHUNKS} chunk={CHUNK} K={NUM_LEVELS} "
        f"trunk_layers={num_layers} mtp_layer={layer_idx} vocab={config.vocab_size} "
        f"cache={weight_cache_path}"
    )

    prompt = list(range(1, TOTAL + NUM_LEVELS + 1))
    logger.info(f"Prompt size is: {len(prompt)}")
    logger.info(f"Prompt is: {prompt}")
    assert max(prompt) < config.vocab_size

    effective_cache_path = weight_cache_path / f"{sp_factor}x{tp_factor}"
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp_factor * tp_factor)
    assert TtPrefillTransformer.check_cache_complete(
        effective_cache_path,
        num_layers,
        experts_per_chip=experts_per_chip,
        first_k_dense=variant.model_config.NUM_DENSE_LAYERS,
    ), f"TTNN cache incomplete for {num_layers} layers at {effective_cache_path}"

    mtp_cache_root = Path(os.getenv(MTP_CACHE_ENV) or weight_cache_path.parent.parent / "glm52_mtp_ttnn_cache")
    mtp_cache_path = mtp_cache_root / f"{variant.name}_{'bh' if is_blackhole() else 'wh'}_{ttnn.get_num_devices()}dev"
    mtp_cache_path = mtp_cache_path / (f"{sp_factor}x{tp_factor}" + (f"_L{num_layers}" if shallow else ""))
    mtp_cache_path = _mtp_cache_dir(mtp_cache_path, Path(ttnn.CONFIG.cache_path) / "glm52_mtp_ttnn_cache")

    init_checker(mtp_cache_path)
    mtp_cached = TtMTPPredictor.check_cache_complete(
        mtp_cache_path,
        layer_idx,
        cache_name_prefix=MTP_CACHE_PREFIX,
        experts_per_chip=experts_per_chip,
        model_cfg=variant.model_config,
    )
    init_checker(effective_cache_path)
    logger.info(
        f"[mtp chunks] MTP layer {layer_idx} cache at {mtp_cache_path}: "
        f"{'present -- loading it' if mtp_cached else 'ABSENT -- building it (first run)'}"
    )

    embed_table = None
    if not skip_pcc:
        embed_table = load_hf_state_dict_filtered(str(model_path), ["model.embed_tokens."])[
            "model.embed_tokens.weight"
        ].to(torch.bfloat16)
        assert list(embed_table.shape) == [
            config.vocab_size,
            config.hidden_size,
        ], f"embedding table {list(embed_table.shape)} != [{config.vocab_size}, {config.hidden_size}]"

    mtp_layer_sd = mtp_layer_state_dict
    mla_weights = mtp_layer_sd["mla_weights"]
    ref_moe_weights = {k: mtp_layer_sd[k] for k in ("gate_weights", "routed_expert_weights", "shared_expert_weights")}

    predictor = TtMTPPredictor(
        mesh_device,
        config,
        variant.model_config,
        {"mtp": mtp_state_dict, "layer": mtp_layer_sd},
        mtp_cfg,
        seq_len=CHUNK,
        num_levels=NUM_LEVELS,
        layer_idx=layer_idx,
        first_cache_slot=num_layers,
        tp_axis=TP_AXIS,
        sp_axis=SP_AXIS,
        num_links=num_links,
        topology=topology,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        dispatch_buffer_capacity_factor=DISPATCH_BUFFER_CAPACITY_FACTOR,
        weight_cache_path=mtp_cache_path,
        cache_name_prefix=MTP_CACHE_PREFIX,
        is_chunked=True,
        max_seq_len=TOTAL,
        slot_num=1,
        layer_num=num_layers + NUM_LEVELS,
    )

    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        weight_cache_path=effective_cache_path,
        num_layers=num_layers,
        seq_len=CHUNK,
        max_seq_len=TOTAL,
        dispatch_buffer_capacity_factor=DISPATCH_BUFFER_CAPACITY_FACTOR,
        num_links=num_links,
        topology=topology,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        is_balanced=False,
        padding_side="right",
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        lm_head_is_column_parallel=True,
        is_chunked=True,
        slot_num=1,
        mtp_predictor=predictor,
    )
    gc.collect()
    ttnn.synchronize_device(mesh_device)

    assert transformer.num_kvpe_cache_layers == num_layers + NUM_LEVELS
    assert transformer.num_mtp_levels == NUM_LEVELS

    ref_mla = None if skip_pcc else [SparseMLAReference(config, mla_weights, seq_len=TOTAL) for _ in range(NUM_LEVELS)]

    # KV dedup: both caches are striped across SP*TP, exactly like the runner allocates them. The
    # sparse indexer always dedups, so an SP-only index cache has no TP axis for it to gather over.
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=TOTAL,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        num_kvpe_cache_layers=transformer.num_kvpe_cache_layers,
        num_users=1,
    )
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=TOTAL,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        num_kvpe_cache_layers=num_full_indexer_layers(config),
        num_users=1,
        dtype=ttnn.bfloat8_b,
    )

    mesh_device.enable_program_cache()

    h0_host: dict = {}
    real_run_mtp = transformer.run_mtp
    derive_gen: dict = {"on": False, "isl": 0, "first": 0}

    def _capture_h0(h_normed, *args, **kwargs):
        h0_host["h"] = None if skip_pcc else _from_device(h_normed, mesh_device)
        if derive_gen["on"] and derive_gen["first"] == 0:
            h0_host["t0"] = _next_token_fn(transformer, derive_gen["isl"])(h_normed)
        return real_run_mtp(h_normed, *args, **kwargs)

    monkeypatch.setattr(transformer, "run_mtp", _capture_h0)

    captured: dict = {}

    def _on_mtp_complete(mtp_out, mtp_generated):
        captured["out"] = mtp_out
        captured["generated"] = list(mtp_generated)

    windows: list[list[list[int]]] = []
    generated: list[int] = []
    logger.info(f"N_MTP = {N_MTP}")
    logger.info(f"CHUNK = {CHUNK}")

    for chunk_idx, (start, actual_isl) in enumerate(CHUNKS):
        stream, real_len, provided = mtp_chunk_stream(prompt, start, CHUNK, N_MTP, NUM_LEVELS, actual_isl=actual_isl)
        actual_end = start + real_len
        logger.info(f"Processing chunk {chunk_idx}")
        logger.info(f"Len(stream) is {len(stream)}")
        logger.info(f"Stream is : {stream} ...")

        assert len(stream) == CHUNK + N_MTP, f"chunk {chunk_idx} stream is {len(stream)}, expected {CHUNK + N_MTP}"
        expected = [MTP_PAD_TOKEN_ID if start + i >= actual_isl else prompt[start + i] for i in range(CHUNK + N_MTP)]
        assert stream == expected, (
            f"chunk {chunk_idx}'s stream must be the prompt slice with pad at every position >= "
            f"actual_isl={actual_isl}; first mismatch at "
            f"{next(i for i, (a, b) in enumerate(zip(stream, expected)) if a != b)}"
        )
        assert_socket_rows(stream, sp_factor, CHUNK, N_MTP)

        scanned = (
            0
            if real_len < CHUNK
            else sum(
                1 for _ in itertools.takewhile(lambda x: x != MTP_PAD_TOKEN_ID, stream[CHUNK : CHUNK + NUM_LEVELS])
            )
        )
        assert scanned == provided, (
            f"chunk {chunk_idx}: scanning the lookahead pad gives {scanned} provided level(s) but the "
            f"arithmetic clamp(actual_isl - actual_end, 0, K) gives {provided}"
        )

        union = _mtp_union(transformer, stream, N_MTP, NUM_LEVELS)
        h0_host.clear()
        captured.clear()
        derive_gen.update(on=(provided < NUM_LEVELS and not skip_pcc), isl=real_len, first=provided)

        logger.info(
            f"[mtp chunks] chunk {chunk_idx}: start={start} real_len={real_len} actual_isl={actual_isl} "
            f"provided={provided}/{NUM_LEVELS}"
        )
        transformer.forward(
            union.trunk,
            kvpe_cache,
            actual_isl=real_len,
            actual_start=start,
            actual_end=actual_end,
            cache_user_id=0,
            index_kv_cache=index_kv_cache,
            mtp_union=union,
            provided_levels=provided,
            input_is_embedded=True,
            on_mtp_complete=_on_mtp_complete,
        )
        ttnn.synchronize_device(mesh_device)
        union.deallocate()

        assert "generated" in captured, "on_mtp_complete never fired -- the MTP branch did not run"
        res = captured["out"]
        assert len(res.x) == NUM_LEVELS
        assert captured["generated"] == [], (
            f"chunk {chunk_idx}: the device path argmaxes, embeds and consumes its generated ids on "
            f"device, so generated_tokens must come back empty on every chunk; got {captured['generated']}"
        )

        assert "h" in h0_host, "run_mtp never ran, so h^0 was never captured"

        if skip_pcc:
            logger.warning(
                f"[mtp chunks] chunk {chunk_idx}: PCC comparison SKIPPED (skip_pcc) -- "
                f"the device ran all {NUM_LEVELS} levels and claim (1) passed, but claims (2)/(3)/(4) "
                "did NOT run. This is not a validated chunk."
            )
            del res
            continue

        seam_is_decisive = True
        generated = []
        if derive_gen["on"]:
            next_token = _next_token_fn(transformer, real_len)
            generated = [
                h0_host["t0"] if k == 0 else next_token(res.out_head_normed[k - 1]) for k in range(provided, NUM_LEVELS)
            ]
            assert len(generated) == NUM_LEVELS - provided
            logger.info(
                f"[mtp chunks] chunk {chunk_idx}: device must have generated {generated} "
                f"for level(s) {list(range(provided, NUM_LEVELS))}"
            )

            dupes = len(generated) - len(set(generated))
            if dupes:
                seam_is_decisive = False
                logger.warning(
                    f"[mtp chunks] chunk {chunk_idx}: generated ids {generated} contain {dupes} "
                    "duplicate(s), so the generation seam below CANNOT distinguish a patch swap "
                    "between the levels that share an id. It still catches a wrong id and a patch "
                    "written outside the generated rows."
                )

        full_seq = prompt[:actual_isl] + generated
        assert len(full_seq) >= actual_end + NUM_LEVELS, (
            f"full_seq is {len(full_seq)} ids; chunk {chunk_idx} level {NUM_LEVELS - 1} indexes up to "
            f"{actual_end + NUM_LEVELS - 1}"
        )
        windows.append([])

        windows[chunk_idx].extend(_expected_window(full_seq, start, level, real_len) for level in range(NUM_LEVELS))
        assert all(len(w) == real_len for w in windows[chunk_idx]), (
            f"chunk {chunk_idx}: a window came out short -- full_seq is {len(full_seq)} ids, which "
            f"does not cover this chunk's shift-{NUM_LEVELS} lookahead"
        )
        host_embeds = [_host_window_embedding(embed_table, w) for w in windows[chunk_idx]]

        dev_x = [_from_device(res.x[k], mesh_device)[:, :, :real_len] for k in range(NUM_LEVELS)]
        dev_out = [_from_device(res.out[k], mesh_device)[:, :, :real_len] for k in range(NUM_LEVELS)]
        dev_normed = [_from_device(res.out_head_normed[k], mesh_device)[:, :, :real_len] for k in range(NUM_LEVELS)]
        del res
        ref_hiddens = [h0_host["h"][:, :, :real_len]] + dev_normed[:-1]

        t0 = time.monotonic()
        ref_xs, ref_outs, ref_normeds, _ = glm_mtp_predictor_reference(
            config,
            mla_weights,
            mtp_state_dict,
            mtp_layer_sd["attn_norm_weight"],
            mtp_layer_sd["ffn_norm_weight"],
            [e.unsqueeze(0) for e in host_embeds],
            ref_hiddens[0].squeeze(0),
            TOTAL,
            moe_weights=ref_moe_weights,
            num_levels=NUM_LEVELS,
            index_share=predictor.index_share,
            hiddens=[h.squeeze(0) for h in ref_hiddens],
            mla_refs=ref_mla,
            actual_start=start,
            actual_end=actual_end,
        )
        logger.info(f"[mtp chunks] chunk {chunk_idx}: CPU reference took {time.monotonic() - t0:.1f}s")

        for level in range(NUM_LEVELS):
            _, msg = assert_with_pcc(ref_xs[level].unsqueeze(0), dev_x[level], FUSED_MTP_PCC)
            logger.info(f"[mtp chunks] chunk {chunk_idx} L{level}: fused projection PCC {msg}")
            _, msg = assert_with_pcc(ref_outs[level].unsqueeze(0), dev_out[level], MTP_MODULE_OUTPUT_PCC)
            logger.info(f"[mtp chunks] chunk {chunk_idx} L{level}: block output PCC {msg}")
            _, msg = assert_with_pcc(ref_normeds[level].unsqueeze(0), dev_normed[level], MTP_MODULE_OUTPUT_PCC)
            logger.info(f"[mtp chunks] chunk {chunk_idx} L{level}: shared_head.norm PCC {msg}")

            if chunk_idx == 0:
                _, msg = assert_with_pcc(ref_xs[level].unsqueeze(0)[:, :, :1], dev_x[level][:, :, :1], FUSED_MTP_PCC)
                logger.info(f"[mtp chunks] chunk {chunk_idx} L{level}: row 0 PCC {msg}")
            if generated and level >= provided:
                seam = real_len - level - 1
                _, msg = assert_with_pcc(
                    ref_xs[level].unsqueeze(0)[:, :, seam:], dev_x[level][:, :, seam:], FUSED_MTP_PCC
                )
                logger.info(
                    f"[mtp chunks] chunk {chunk_idx} L{level}: generation seam PCC {msg}"
                    f"{'' if seam_is_decisive else ' (duplicate ids -- swap-blind, see warning above)'}"
                )

        del dev_x, dev_out, dev_normed, ref_hiddens, ref_xs, ref_outs, ref_normeds

    logger.info(f"[mtp chunks] generated tokens for the final K slots: {generated}")
