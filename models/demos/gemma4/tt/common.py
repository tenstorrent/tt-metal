# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 model creation utility — matches tt_transformers interface.

Usage:
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device, max_batch_size=1, max_seq_len=8192,
    )
"""

import os

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.common.weight_cache import (
    build_cached_state_dict,
    checkpoint_name,
    mark_weight_cache_complete,
    weight_cache_is_complete,
)
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.assistant.model import Gemma4AssistantModel
from models.demos.gemma4.tt.ccl import CCLManager, mesh_tp_axis
from models.demos.gemma4.tt.mesh_orientation import dispatch_prefill_default
from models.demos.gemma4.tt.model import Gemma4Model
from models.demos.gemma4.tt.model_config import Gemma4AssistantArgs, Gemma4ModelArgs
from models.demos.gemma4.tt.precision import Gemma4Precision

# Weights gemma4 consumes on the HOST (not just via ttnn.as_tensor) and that therefore must be
# loaded for real even on a warm cache (see #45400 follow-up analysis of models/demos/gemma4/tt):
#  - token embedding: F.embedding(tokens, _embed_weight_cpu)      (model.py:1218/1238/1421)
#  - per-layer-input embed/proj/norm (E2B/E4B):                    (model.py:615-635)
#  - per-layer learned scalar read via .item():                   (layer.py:122-123)
# Everything else flows through ttnn.as_tensor(cache_file_name=...) and is placeholder-safe.
_GEMMA4_HOST_WEIGHT_SUFFIXES = (
    "embed_tokens.weight",
    "embed_tokens_per_layer.weight",
    "per_layer_model_projection.weight",
    "per_layer_projection_norm.weight",
    ".layer_scalar",
)


def _gemma4_is_host_weight(key):
    return any(key.endswith(s) for s in _GEMMA4_HOST_WEIGHT_SUFFIXES)


def gemma4_env_flag(name, default="0"):
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def gemma4_cache_layout(mesh_device, mesh_config):
    """Every build knob that decides WHICH tensorbin FILENAMES a build needs (the warm-cache identity).

    A warm marker seeded before a filename change certifies a build whose files do not exist yet, and
    as_tensor then persists dataless placeholders under the new names (weight_cache.py "Build options
    that change an as_tensor cache FILENAME ... must match exactly"). #50648 renamed the fused gate/up,
    .ws attention and .ws down-proj files without touching this identity and poisoned the Gemma-4-E4B
    caches on bh_p300 / bh_quietbox_2 (#54831: 2085 of 2131 keys served as torch.empty and written to
    disk). Bump the layout tag whenever a cache filename scheme changes.
    """
    from models.demos.gemma4.tt.experts.weights import _expert_parallel_factors

    ep, tp, _ep_axis, tp_axis = _expert_parallel_factors(mesh_device, mesh_config)
    return {
        # v3: routed experts are stored whole per chip (*_ep{ep}tp{tp}_whole_* fused [up|gate]) instead
        # of intermediate-sliced; the v2 marker certified only the old expert filenames.
        "layout": "v3-experts-whole-ep-tp",
        # Fractured builds write *_f8x4_* MLP files; a marker seeded by a plain
        # TP build must not certify them.
        "weight_layout": (
            f"fractured-tp_axis{getattr(mesh_config, 'tp_axis', 1)}"
            if bool(getattr(mesh_config, "weight_fracture", False))
            else "tp"
        ),
        # The routed-expert files are named by their partition (*_ep{ep}tp{tp}_whole_*), and
        # GEMMA4_MOE_EP changes the partition on the same mesh: a marker seeded under one
        # partition must not certify the files of another.
        "expert_partition": f"ep{ep}tp{tp}_tp_axis{tp_axis}",
        "attn_dram_shard": os.environ.get("GEMMA4_ATTN_DRAM_SHARD", "1"),
        "mlp_dram_shard": os.environ.get("GEMMA4_MLP_DRAM_SHARD", "1"),
        "dram_cores": os.environ.get("GEMMA4_DRAM_CORES", "8"),
        # The token-dispatch prefill converts the routed experts into the fused kernel's own cache files; a
        # marker seeded without them must not certify a build that needs them (the effective value: the path is
        # the default on a Blackhole (4,1) column mesh, see mesh_orientation.dispatch_prefill_default).
        "moe_dispatch_prefill": "1"
        if dispatch_prefill_default(getattr(mesh_device, "shape", None), blackhole=is_blackhole())
        else "0",
        # The local routed prefill (experts/local_routed_prefill.py) loads the routed experts into the kernel's
        # per-expert cache files too (layer_<i>.local_routed_<rows>x<cols>.*).
        "moe_local_routed_prefill": os.environ.get("GEMMA4_MOE_LOCAL_ROUTED_PREFILL", "0"),
    }


# Generator prefill chunk used with CP prefill: 6144 rows per column fill the SDPA grid.
GEMMA4_CP_PREFILL_CHUNK = 24576


def gemma4_cp_prefill_engaged(mesh_device):
    """True only when ``create_tt_model`` will actually set ``mesh_config.cp_prefill``.

    Prefill-chunk defaults use this so GEMMA4_CP_PREFILL=1 without fracture cannot select the large chunk.
    """
    shape = getattr(mesh_device, "shape", None)
    try:
        is2d = shape is not None and shape[0] > 1 and shape[1] > 1
    except (TypeError, IndexError):
        is2d = False
    return (
        gemma4_env_flag("GEMMA4_CP_PREFILL")
        and gemma4_env_flag("GEMMA4_GALAXY_FRACTURE")
        and is2d
        and not gemma4_env_flag("GEMMA4_GALAXY_LANES")
    )


def _moe_expert_parallel(hf_text_config, mesh_shape, tp_axis) -> int:
    """Expert-parallel degree for a MoE checkpoint on a 2-D mesh: the size of the non-TP axis
    (each row of it computes its own slice of the expert set; see experts/__init__.py).
    1 for dense checkpoints, 1-D meshes, or GEMMA4_MOE_EP=0."""
    if not (getattr(hf_text_config, "num_experts", None) or 0):
        return 1
    if os.environ.get("GEMMA4_MOE_EP", "1").lower() in ("0", "false", "no"):
        return 1
    ep_axis = 0 if tp_axis == 1 else 1
    return int(mesh_shape[ep_axis]) if int(mesh_shape[ep_axis]) > 1 else 1


def create_tt_model(
    mesh_device,
    max_batch_size=1,
    max_seq_len=8192,
    dtype=ttnn.bfloat16,
    state_dict=None,
    num_layers=None,
    mesh_config=None,
    paged_attention_config=None,
    create_kv_cache=True,
    model_path=None,
    bounded_sliding_kv_cache: bool = False,
):
    """
    Create Gemma4 model with all weights loaded to device.

    Returns:
        (model_args, model, tt_kv_cache, state_dict)
    """
    model_path = (
        model_path
        or os.getenv("HF_MODEL")
        or os.getenv("GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it")
    )

    hf_config = Gemma4ModelArgs.load_hf_config(model_path)
    model_args = Gemma4ModelArgs.from_hf_config(hf_config)
    model_args.model_cache_path = model_args.resolve_model_cache_path(model_path)
    # Store the real HF text config for RoPE creation (Gemma4TextRotaryEmbedding needs it)
    hf_text_config = getattr(hf_config, "text_config", hf_config)
    model_args._hf_text_config = hf_text_config

    if num_layers is not None:
        model_args.num_hidden_layers = num_layers

    if mesh_config is None:
        is_mesh = hasattr(mesh_device, "shape")
        num_devices = mesh_device.get_num_devices() if is_mesh else 1
        _fracture = gemma4_env_flag("GEMMA4_GALAXY_FRACTURE")
        _lanes = gemma4_env_flag("GEMMA4_GALAXY_LANES")
        if is_mesh and num_devices > 1 and _fracture and mesh_device.shape[0] > 1 and mesh_device.shape[1] > 1:
            # Galaxy one-instance: 2D-fractured weights with TP (heads) on the size-8 axis; (8,4) and (4,8)
            # are both valid, so orientation stays a MESH_DEVICE knob. GEMMA4_GALAXY_LANES=1 lane-shards the batch.
            _shape = tuple(mesh_device.shape)
            _tp_axis = 0 if _shape[0] == 8 else (1 if _shape[1] == 8 else 0)
            _ep = _moe_expert_parallel(hf_text_config, _shape, _tp_axis)
            mesh_config = MeshConfig(
                mesh_device.shape,
                decode=ModeConfig(tp=_shape[_tp_axis], ep=_ep),
                prefill=ModeConfig(tp=_shape[_tp_axis], ep=_ep),
                tp_axis=_tp_axis,
                weight_fracture=True,
            )
            mesh_config.lane_sharded = _lanes
            mesh_config.cp_prefill = gemma4_env_flag("GEMMA4_CP_PREFILL")
            if mesh_config.lane_sharded and mesh_config.cp_prefill:
                # Lane prefill hands the real page table only to the owner column, while CP partitions Q
                # across every column; lanes win until CP gathers KV to the owner lane first.
                logger.warning("GEMMA4_CP_PREFILL ignored: incompatible with GEMMA4_GALAXY_LANES (owner-lane KV)")
                mesh_config.cp_prefill = False
        elif is_mesh and num_devices > 1:
            # Plain TP over one mesh axis (GEMMA4_TP_AXIS; default the columns, the rows on a column mesh); the
            # other axis is EP. On a column mesh (the 4-chip line opened as MeshShape(4, 1)) the TP group sits on
            # the rows, where the token-dispatch prefill's dispatch/combine ops can run (they need axis 0).
            _tp_axis = mesh_tp_axis(tuple(mesh_device.shape))
            _tp = int(mesh_device.shape[_tp_axis])
            _ep = _moe_expert_parallel(hf_text_config, tuple(mesh_device.shape), _tp_axis)
            mesh_config = MeshConfig(
                mesh_device.shape,
                decode=ModeConfig(tp=_tp, ep=_ep),
                prefill=ModeConfig(tp=_tp, ep=_ep),
                tp_axis=_tp_axis,
            )
            if _tp_axis == 0:
                logger.info(
                    f"Gemma4 mesh {tuple(mesh_device.shape)}: TP={_tp} over mesh axis 0 (GEMMA4_TP_AXIS=0), EP={_ep}"
                )
            elif int(mesh_device.shape[1]) == 1:
                logger.warning(
                    f"Gemma4 mesh {tuple(mesh_device.shape)} is a column mesh but the TP group is on axis 1: "
                    f"TP={_tp}, EP={_ep} (the dense weights are replicated on every chip); set GEMMA4_TP_AXIS=0 "
                    "for TP over the rows"
                )
        else:
            mesh_config = MeshConfig((1, 1), decode=ModeConfig(tp=1))

    is_mesh = hasattr(mesh_device, "shape")
    num_devices = mesh_device.get_num_devices() if is_mesh else 1
    if is_mesh and num_devices > 1:
        # num_links=None -> arch default (2 on Blackhole) so the per-layer TP
        # all-reduces (the dominant ~31% of prefill device time) use full
        # inter-device bandwidth.
        ccl_manager = CCLManager(mesh_device)
    else:
        ccl_manager = None

    # Warm ttnn cache => skip the full HF weight load and build from .tensorbin. Hybrid: the few
    # host-consumed weights (token embedding, per-layer scalars/PLI) are served real from the
    # sidecar, the rest as dataless placeholders. Generalizes PR #50550 to gemma4 (#45400).
    # Qualify the cache by mesh geometry BEFORE resolving cache_dir: ttnn.as_tensor
    # reloads tensorbins as-is and ignores mesh_mapper, so a TP=4 cache built on
    # MeshShape([2,4]) must not be reused on [1,4] (QB2). Setting cluster_shape here
    # keeps the warm-cache marker (cache_dir) and the tensorbin path on the same
    # directory instead of letting them diverge.
    _worker_mesh = tuple(mesh_device.shape) if hasattr(mesh_device, "shape") else (1, 1)
    model_args.cluster_shape = _worker_mesh
    cache_dir = model_args.weight_cache_path(dtype)
    # Resolved early so it can key the cache identity: gemma4 embeds each module's dtype in its
    # tensorbin FILENAME (attention/experts/shared_mlp/router *_{dtype} suffixes), so an edit to
    # precision_overrides.json changes which files a build needs. Without the precision in the
    # variant, a marker seeded under the old overrides would certify a warm build whose files do
    # not exist -- and as_tensor would persist placeholders for them. (#45400 review, finding B2)
    # NOTE: use the SERVED max_seq_len argument, not model_args.max_seq_len --
    # model_args still carries its construction default here (the served value is
    # applied further below), so reading it silently skipped the bfp8 context
    # ceiling at long context.
    _precision_for_variant = Gemma4Precision.load(model_path, _worker_mesh, max_seq_len=max_seq_len)
    # The variant must also pin every knob that decides WHICH tensorbin FILENAMES a build needs, not
    # only their dtype (see gemma4_cache_layout).
    _cache_layout = gemma4_cache_layout(mesh_device, mesh_config)
    cache_identity = dict(
        # vLLM hands over the resolved hub snapshot dir, not the HF id: keyed on the plain
        # basename the marker seeded by the e2e demo never matched and every server start
        # cold-loaded the full HF checkpoint (31B on QB2: ~12 of the 20 budgeted minutes).
        model_name=checkpoint_name(model_path) or "gemma4",
        n_layers=model_args.num_hidden_layers,
        mesh_shape=_worker_mesh,
        build_variant={
            "precision": {k: str(v) for k, v in sorted(_precision_for_variant._overrides.items())},
            "cache_layout": _cache_layout,
        },
    )
    loaded_real_weights = False
    # The decode MoE kernel uploads its bfp4 expert weights from the host tensors on every build (no cache yet),
    # so a warm-cache placeholder cannot serve it: always load the checkpoint when it is on.
    needs_host_weights = os.environ.get("GEMMA4_MOE_DECODE_KERNEL", "0") == "1"
    if state_dict is None:
        if num_layers is None and not needs_host_weights and weight_cache_is_complete(cache_dir, **cache_identity):
            logger.info("Warm ttnn weight cache detected -- skipping HF state_dict load (gemma4 hybrid).")
            state_dict = build_cached_state_dict(
                cache_dir, args=model_args, build_variant=cache_identity["build_variant"]
            )
        else:
            state_dict = Gemma4ModelArgs.load_state_dict(model_path, dummy_weights=False)
            loaded_real_weights = bool(state_dict)

    tensor_cache_path = str(cache_dir)

    # Per-module dtype overrides from precision_overrides.json, resolved once
    # above so the cache identity and the model share one value.
    precision = _precision_for_variant

    # A placeholder state_dict holds no weight data, so as_tensor must raise on an unloadable cache
    # file rather than rebuild it from the placeholder.
    with ttnn.as_tensor_cache_repair(not getattr(state_dict, "is_placeholder", False)):
        model = Gemma4Model(
            mesh_device=mesh_device,
            hf_config=model_args,
            state_dict=state_dict,
            ccl_manager=ccl_manager,
            dtype=dtype,
            tensor_cache_path=tensor_cache_path,
            mesh_config=mesh_config,
            max_seq_len=max_seq_len,
            max_local_batch_size=max_batch_size,
            num_layers=num_layers,
            paged_attention_config=paged_attention_config,
            create_kv_cache=create_kv_cache,
            precision=precision,
            bounded_sliding_kv_cache=bounded_sliding_kv_cache,
        )

    # After a full cold build, record completion (+ capture host-consumed weights to the sidecar)
    # so future runs can skip the HF load.
    if loaded_real_weights and num_layers is None:
        mark_weight_cache_complete(cache_dir, state_dict, is_host_weight=_gemma4_is_host_weight, **cache_identity)

    _log_dram_usage(mesh_device, "after weight load")
    return model_args, model, model.tt_kv_cache, state_dict


def _log_dram_usage(mesh_device, when: str) -> None:
    """One line of DRAM usage (first device, per bank and x8 banks) so the weight budget is visible in the boot log:
    the (4,1) column-mesh build carries a second, bfp8 copy of the routed experts for the dispatch prefill."""
    try:
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        alloc = int(view.total_bytes_allocated_per_bank)
        free = getattr(view, "total_bytes_free_per_bank", None)
        total = getattr(view, "total_bytes_allocatable_per_bank", None)
        banks = 8
        msg = f"Gemma4 DRAM {when}: {alloc / 1e9:.2f} GB per bank allocated (~{banks * alloc / 1e9:.1f} GB per chip)"
        if free is not None:
            msg += f", {int(free) / 1e9:.2f} GB per bank free"
        if total is not None:
            msg += f" of {int(total) / 1e9:.2f} GB"
        logger.info(msg)
    except Exception as exc:  # noqa: BLE001 -- a log line must never fail the build
        logger.debug(f"Gemma4 DRAM usage unavailable ({exc})")


def create_assistant_model(
    mesh_device,
    target_model,
    mesh_config,
    ccl_manager,
    dtype=ttnn.bfloat16,
    assistant_path=None,
    state_dict=None,
    max_local_batch_size=1,
    bounded_sliding_kv_cache=None,
    max_seq_len=None,
):
    """Create the Gemma4 it-assistant drafter, sharing the target's mesh/CCL.

    The drafter cross-attends into ``target_model``'s KV caches and reuses its
    RoPE caches + raw token embedding, so it must be built from the same target
    instance used for decoding.

    Returns:
        (assistant_args, assistant_model)
    """
    assistant_path = assistant_path or os.getenv("GEMMA4_ASSISTANT_MODEL")
    if not assistant_path:
        raise ValueError(
            "No assistant model path. Set GEMMA4_ASSISTANT_MODEL (e.g. google/gemma-4-31B-it-assistant) "
            "or pass assistant_path=."
        )

    hf_config = Gemma4AssistantArgs.load_hf_config(assistant_path)
    assistant_args = Gemma4AssistantArgs.from_hf_config(hf_config)
    assistant_args.model_cache_path = assistant_args.resolve_model_cache_path(assistant_path)

    if assistant_args.backbone_hidden_size != target_model.hidden_size:
        raise ValueError(
            f"Assistant backbone_hidden_size ({assistant_args.backbone_hidden_size}) != target hidden_size "
            f"({target_model.hidden_size}). The assistant must match its target model."
        )
    # Bounded target KV is supported now: the drafter's attention configs take
    # the same cache_position_modulo as the target's sliding layers (see the
    # bounded_sliding_kv_cache plumbing below and assistant/model.py), so its
    # cross-attention wraps absolute positions into the same ring. This is what
    # makes >=128k spec decode reachable on 31B, where unbounded KV does not fit.
    # It requires the ring sizes to agree; the assistant's own sliding_window
    # matches its target's (1024 on both 12B and 31B), so verify that here
    # rather than assuming it.
    # GATE LIFTED. The clobbering described here was real -- verify writes all K+1
    # candidates up front, and in a ring of EXACTLY the window, slot (p+j)%W still
    # holds live position p+j-W. It is fixed by ring HEADROOM (the spec path runs
    # ring = 2*window, so speculative slots fall outside the window; see
    # attention.bounded_ring_modulo), plus the last-chunk expansion threshold fix
    # in Gemma4Generator._expand_bounded_last_chunk.
    # Measured 31B @ 32k, greedy/traced: bounded 2.40/5 @ 42.26 tok/s/u vs
    # unbounded 2.40/5 @ 42.42 -- parity, matching text. 128k 2.78/5 @ 36.08
    # (baseline 24.44); 256k 1.70/5 @ 16.35 (baseline 16.97 -- coherent but spec
    # is NOT a win at 256k: verify cost scales with context, acceptance falls).
    if getattr(target_model, "bounded_sliding_kv_cache", False):
        _tgt_win = getattr(target_model, "sliding_window", None) or getattr(
            getattr(target_model, "hf_config", None), "sliding_window", None
        )
        _asst_win = getattr(assistant_args.text_args, "sliding_window", None)
        if _tgt_win is not None and _asst_win is not None and int(_tgt_win) != int(_asst_win):
            raise NotImplementedError(
                f"Bounded spec decode needs matching sliding windows: target {_tgt_win} vs "
                f"assistant {_asst_win}. The drafter cross-attends the target's bounded ring, "
                "so a different window would wrap positions to the wrong slots."
            )

    if state_dict is None:
        state_dict = Gemma4AssistantArgs.load_state_dict(assistant_path, dummy_weights=False)

    mesh_shape = tuple(mesh_device.shape) if hasattr(mesh_device, "shape") else (1, 1)
    assistant_args.cluster_shape = mesh_shape
    # Serve length: Gemma4AssistantArgs.max_seq_len DEFAULTS to 131072, and the
    # drafter's layers are built from it (assistant/model.py max_seq_len=...). Left
    # unset, a 256k run builds the drafter for HALF the context while positions run
    # to 262143 -- the drafter then produces noise and acceptance collapses to
    # ~0.00/5 (measured), while <=128k looked fine because the default covered it.
    if max_seq_len is not None:
        assistant_args.max_seq_len = int(max_seq_len)
        if hasattr(assistant_args, "text_args"):
            assistant_args.text_args.max_seq_len = int(max_seq_len)
    tensor_cache_path = str(assistant_args.weight_cache_path(dtype, mesh_shape=mesh_shape))

    model = Gemma4AssistantModel(
        mesh_device=mesh_device,
        assistant_args=assistant_args,
        target_model=target_model,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        dtype=dtype,
        tensor_cache_path=tensor_cache_path,
        mesh_config=mesh_config,
        max_local_batch_size=max_local_batch_size,
        # Match the TARGET's KV mode: with bounded sliding caches the drafter's
        # cross-attention must wrap positions into the same ring. Inferred from
        # the target when not stated explicitly.
        # Whether the DRAFTER wraps positions must match the caches it actually
        # reads, not the target's global mode. The drafter cross-attends only the
        # LAST layer of each type; full-attention layers are always unbounded, and
        # the last sliding layer is EXEMPTED from bounding for exactly this reason
        # (Gemma4Model._spec_unbounded_layer). So when that exemption is active,
        # both caches the drafter touches hold absolute positions and it must NOT
        # apply the ring modulo -- otherwise it looks up p % window in a
        # full-length cache and drafts noise (measured: acceptance 0.12/5 at 128k
        # bounded vs 2.78/5 unbounded, 0.00/5 at 256k).
        bounded_sliding_kv_cache=(
            bounded_sliding_kv_cache
            if bounded_sliding_kv_cache is not None
            else (
                bool(getattr(target_model, "bounded_sliding_kv_cache", False))
                and getattr(target_model, "_spec_unbounded_layer", None) is None
            )
        ),
    )
    return assistant_args, model
