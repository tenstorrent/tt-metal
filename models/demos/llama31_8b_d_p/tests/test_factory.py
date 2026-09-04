# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared test scaffolding for the `llama31_8b_d_p` prefill bring-up.

Mirrors `models/demos/minimax_m3/tests/test_factory.py` (`TestFactory` at :45,
`setup_test` at :56, `minimax_config_dims` at :25, `requires_hf_reference` at :35).

Three responsibilities, in increasing order of what they need:

1. `llama_config_dims()` — dimensions only, from the bundled `configs/<Name>/config.json`.
   **No HuggingFace, no network, no checkpoint, no device.**
2. `requires_hf_reference` — a `skipif` marker for the tests that genuinely need real weights.
   See `bringup_log/07_RISKS.md` R-003: nothing is staged on the bring-up machine, so those
   tests skip rather than fail.
3. `TestFactory.setup_test()` — builds `MeshConfig` + `CCLManager` on a real mesh. Its imports
   are deliberately *inside the function* because `tt/config.py` and `tt/ccl.py` are P5
   deliverables; this keeps module import cheap and side-effect-free today
   (`bringup_log/07_RISKS.md` R-011).

HF anchor: `transformers.models.llama.configuration_llama.LlamaConfig`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# --- Model identity -------------------------------------------------------------------
# DEC-001: `llama31_8b` resolves to Llama-3.1-8B-Instruct. There is no public "Llama-3.2 8B";
# the in-tree Llama-3.2 family is 1B/3B text and 11B/90B Vision. See bringup_log/00_MODEL_CARD.md
# section 1 and bringup_log/07_RISKS.md R-001 (the one assumption the user must confirm).
MODEL_NAME = "Llama-3.1-8B-Instruct"

_PKG_ROOT = Path(__file__).resolve().parent.parent
CONFIG_JSON = _PKG_ROOT / "configs" / MODEL_NAME / "config.json"

# DEC-005: the bundled config is a verbatim copy of the tt_transformers one, and the
# reference test asserts byte-identity so the copy cannot silently drift.
UPSTREAM_CONFIG_JSON = _PKG_ROOT.parent.parent / "tt_transformers" / "model_params" / MODEL_NAME / "config.json"


def llama_config_dims() -> dict:
    """Return the raw `config.json` dict, plus the two values Llama derives rather than stores.

    Reads the bundled config only — no `transformers`, no network, no checkpoint. This is what
    dimension-only tests should use.

    Derived keys added:
      * `head_dim`      — absent from the Llama-3.1-8B config; `hidden_size // num_attention_heads`.
                          HF derives it identically (`configuration_llama.py:87-88`).
      * `gqa_group_size`— `num_attention_heads // num_key_value_heads` (= 4).

    `rope_theta` is read from the raw JSON where it is still top-level. Do **not** switch this to
    `LlamaConfig.rope_theta`: under transformers 5.12.1 that attribute **does not exist at all**
    (it raises `AttributeError`, and `getattr(cfg, "rope_theta", DEFAULT)` silently returns
    `DEFAULT`), because the value moved into `rope_parameters`. Measured in P5.1; this supersedes
    the "exists and is None" wording of R-002 (see BRINGUP_RECIPE.md Appendix F.2 and
    `models/tt_transformers/tt/common.py:165` `get_rope_theta`).
    """
    with open(CONFIG_JSON) as f:
        cfg = json.load(f)

    if "head_dim" not in cfg:
        assert cfg["hidden_size"] % cfg["num_attention_heads"] == 0, (
            f"hidden_size {cfg['hidden_size']} not divisible by num_attention_heads "
            f"{cfg['num_attention_heads']}; head_dim cannot be derived"
        )
        cfg["head_dim"] = cfg["hidden_size"] // cfg["num_attention_heads"]

    assert cfg["num_attention_heads"] % cfg["num_key_value_heads"] == 0, (
        f"num_attention_heads {cfg['num_attention_heads']} not divisible by num_key_value_heads "
        f"{cfg['num_key_value_heads']}; this is not a valid GQA shape"
    )
    cfg["gqa_group_size"] = cfg["num_attention_heads"] // cfg["num_key_value_heads"]

    return cfg


def rope_theta(cfg: dict) -> float:
    """Extract RoPE theta from a raw config dict, transformers-5.x-safe.

    Delegates to `models/tt_transformers/tt/common.py:165` `get_rope_theta`, which checks
    top-level `rope_theta`, then `rope_parameters["rope_theta"]` (flat: Qwen/Llama), then
    `rope_parameters["full_attention"]["rope_theta"]` (Gemma-style). Asserts non-None so a
    silently-missing theta fails loud instead of producing a wrong RoPE at every position
    (bringup_log/07_RISKS.md R-002).
    """
    from models.tt_transformers.tt.common import get_rope_theta

    theta = get_rope_theta(cfg)
    assert theta is not None, "rope_theta resolved to None; refusing to build a RoPE table"
    return float(theta)


def rope_scaling(cfg: dict) -> dict:
    """Return the llama3 RoPE-scaling parameters, asserting the two that the repo hard-codes.

    `models/tt_transformers/tt/common.py:405` `compute_llama3_parameters(freqs, scale_factor,
    orig_context_len)` hard-codes `low_freq_factor = 1` (:407) and `high_freq_factor = 4` (:408) —
    it does *not* read them from the config, contrary to BRINGUP_RECIPE.md:620-624. Asserting them
    here converts a silent-wrong into a loud-fail for any config that disagrees.
    See bringup_log/07_RISKS.md R-006.
    """
    rs = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
    assert rs.get("rope_type") == "llama3", f"expected rope_type 'llama3', got {rs.get('rope_type')!r}"
    assert float(rs["low_freq_factor"]) == 1.0, (
        f"low_freq_factor is {rs['low_freq_factor']}, but "
        f"models/tt_transformers/tt/common.py:407 hard-codes 1 and would silently ignore it"
    )
    assert float(rs["high_freq_factor"]) == 4.0, (
        f"high_freq_factor is {rs['high_freq_factor']}, but "
        f"models/tt_transformers/tt/common.py:408 hard-codes 4 and would silently ignore it"
    )
    return rs


# --- Real-weight gating ---------------------------------------------------------------
_HF_MODEL = os.getenv("HF_MODEL")

requires_hf_reference = pytest.mark.skipif(
    not (_HF_MODEL and os.path.isdir(_HF_MODEL)),
    reason="HF_MODEL is not a directory; real-weight reference unavailable (see 07_RISKS.md R-003)",
)


def hf_model_path() -> str:
    """The staged checkpoint directory. Only call under `requires_hf_reference`."""
    assert _HF_MODEL and os.path.isdir(_HF_MODEL), "HF_MODEL is not a directory"
    return _HF_MODEL


# --- Device scaffolding ---------------------------------------------------------------
class TestFactory:
    """Builds the per-test device objects once, so unit tests stay about the model."""

    # DEC-002: the validated deployment target on the 32-device Blackhole Galaxy.
    # mesh_shape is (SP, TP) -- models/demos/common/prefill/adapter.py:57.
    TARGET_MESH_SHAPE = (4, 8)
    TARGET_TP = 8
    TARGET_SP = 4

    # Single-card bring-up shape for P5-P7 (BRINGUP_RECIPE.md:578).
    SINGLE_CARD_MESH_SHAPE = (1, 1)

    MESH_SHAPES = {
        "1x1": (1, 1),
        "1x2": (1, 2),
        "1x4": (1, 4),
        "1x8": (1, 8),
        "4x8": (4, 8),
    }

    # Sequence lengths used by the module gates (BRINGUP_RECIPE.md:616, :663, :690).
    NORM_SEQ_LENS = (32, 512, 4096)
    ATTN_SEQ_LENS = (128, 512, 2048)

    @staticmethod
    def setup_submesh(parent_mesh, submesh_shape, *, tp=None, weight_dtype=None, tensor_cache_path=None):
        """`setup_test` on a **submesh** carved from the open galaxy (`DEC-080`, `DEC-081`).

        `parent_mesh` is the full `(4, 8)` mesh the `mesh_device` fixture opened; `submesh_shape` is
        the shape this case wants. A top-level partial mesh cannot bring the fabric up on this box
        (measured: `tests/fabric_topology_matrix.py`), so every multi-device shape below the galaxy
        is a submesh of it. The topology is **always** `Topology.Ring`: `Topology.Linear` hangs the
        machine on the 8-wide logical row, which spans two physical rows of the `MeshShape([8, 4])`
        system mesh (`DEC-081`, superseding `DEC-020`'s topology column).

        Returns the same dict as `setup_test`, with `mesh_device` set to the **submesh** — hand that
        to the module under test, never the parent. Submeshes are kept alive by the parent and closed
        by the `mesh_device` fixture's `get_submeshes()` loop (repo `conftest.py:669`), so a caller
        must not close them.
        """
        import ttnn
        from models.demos.llama31_8b_d_p.tt.ccl import CCLManager
        from models.demos.llama31_8b_d_p.tt.config import MeshConfig
        from models.demos.llama31_8b_d_p.tt.model_config import llama_hf_config
        from models.demos.llama31_8b_d_p.utils.general_utils import get_default_num_links

        rows, cols = tuple(submesh_shape)
        assert tuple(parent_mesh.shape) == TestFactory.TARGET_MESH_SHAPE, (
            f"setup_submesh expects the full galaxy {TestFactory.TARGET_MESH_SHAPE} as the parent, "
            f"got {tuple(parent_mesh.shape)}"
        )
        if (rows, cols) == TestFactory.TARGET_MESH_SHAPE:
            submesh = parent_mesh
        else:
            submesh = parent_mesh.create_submesh(ttnn.MeshShape(rows, cols), ttnn.MeshCoordinate(0, 0))

        mesh_config = MeshConfig((rows, cols), tp=cols if tp is None else tp)
        ccl_manager = CCLManager(
            submesh,
            num_links=get_default_num_links(submesh),
            topology=ttnn.Topology.Ring,
        )
        return {
            "mesh_device": submesh,
            "parent_mesh": parent_mesh,
            "mesh_shape": (rows, cols),
            "mesh_config": mesh_config,
            "ccl_manager": ccl_manager,
            "hf_config": llama_hf_config(llama_config_dims()),
            "weight_dtype": weight_dtype if weight_dtype is not None else ttnn.bfloat16,
            "tensor_cache_path": tensor_cache_path,
        }

    @staticmethod
    def setup_test(mesh_device, *, tp: int | None = None, weight_dtype=None, tensor_cache_path=None):
        """Build `MeshConfig` + `CCLManager` + the normalised `hf_config` and return them in a dict.

        The returned `hf_config` is a `LlamaHFConfig` object (`DEC-009`), so a test can hand it
        straight to a `tt/` module. Tests that need the raw dict (e.g. to build a `LlamaConfig`
        reference) call `llama_config_dims()` themselves.

        Imports are function-local on purpose: `tt/config.py` and `tt/ccl.py` are P5 deliverables,
        and this module must stay importable (and cheap) before they exist. Calling this before P5
        raises `ModuleNotFoundError` naming the missing module. See 07_RISKS.md R-011.
        """
        import ttnn
        from models.demos.llama31_8b_d_p.tt.ccl import CCLManager
        from models.demos.llama31_8b_d_p.tt.config import MeshConfig
        from models.demos.llama31_8b_d_p.tt.model_config import llama_hf_config
        from models.demos.llama31_8b_d_p.utils.general_utils import get_default_num_links

        mesh_shape = tuple(mesh_device.shape)
        if tp is None:
            tp = mesh_shape[1]

        mesh_config = MeshConfig(mesh_shape, tp=tp)
        ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))

        return {
            "mesh_device": mesh_device,
            "mesh_shape": mesh_shape,
            "mesh_config": mesh_config,
            "ccl_manager": ccl_manager,
            # DEC-009: modules take the normalised OBJECT, never the raw dict. `llama_config_dims()`
            # is the dict form and stays available for dimension-only / dict-taking helpers.
            "hf_config": llama_hf_config(llama_config_dims()),
            "weight_dtype": weight_dtype if weight_dtype is not None else ttnn.bfloat16,
            "tensor_cache_path": tensor_cache_path,
        }


# --- Multi-device parametrisation (P8 / `DEC-080`, `DEC-081`) --------------------------
def parametrize_galaxy_submeshes(submesh_shapes, *, trace_region_size=None):
    """Parametrise `(mesh_device, device_params, submesh_shape)` for the P8 multi-device gates.

    Every case opens the **same** parent — the full `(4, 8)` galaxy with `FABRIC_1D_RING` — and
    carries the shape the test should carve out of it with `TestFactory.setup_submesh`. The test body
    therefore never opens a mesh, and the fabric is brought up once for the whole 32-device cluster,
    which is the only way it comes up at all on this machine (`DEC-080`).

    Contrast `models/demos/minimax_m3/tests/test_factory.py:89` `parametrize_mesh_with_fabric`, which
    parametrises `mesh_device` with the *sub*-shape and lets the fixture open it directly. That is the
    right shape for a LoudBox, where `(1,8)` is the whole machine; here it dies in fabric bring-up.

    Case ids are the submesh shape (`1x2`, `1x8`, `2x8`, `4x8`), so `pytest -k 2x8` filters cleanly.
    Shapes that do not fit inside the galaxy are dropped.
    """
    import ttnn

    device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}
    if trace_region_size is not None:
        device_params["trace_region_size"] = trace_region_size
    rows, cols = TestFactory.TARGET_MESH_SHAPE
    shapes = [tuple(s) for s in submesh_shapes if s[0] <= rows and s[1] <= cols]
    params = [
        pytest.param(TestFactory.TARGET_MESH_SHAPE, dict(device_params), shape, id=f"{shape[0]}x{shape[1]}")
        for shape in shapes
    ]
    if not params:
        params = [
            pytest.param(
                TestFactory.TARGET_MESH_SHAPE,
                dict(device_params),
                TestFactory.TARGET_MESH_SHAPE,
                id="none",
                marks=pytest.mark.skip(reason="no requested submesh shape fits in the (4,8) galaxy"),
            )
        ]
    return pytest.mark.parametrize(
        "mesh_device, device_params, submesh_shape", params, indirect=["mesh_device", "device_params"]
    )


# --- Promoted numerical helpers (P6 / `DEC-046`) ---------------------------------------
# `DEC-037` parked `quantize_like_device` and `err_ratio` in `tests/unit/test_mlp_vs_ref.py` (at
# lines 67 and 78 as of P5) because `test_factory.py` was being edited by a concurrent session. They
# are now used by eight gates (`G-MLP`, `G-ATTN`, `G-KV`, `G-LAYER`, `G-WEIGHTS`, `G-MODEL`, and
# P9's `G-CLEAN` additions), so this is their home.
#
# P9 finished the move (`DEC-124`): the P5 copies in `test_mlp_vs_ref.py` are **deleted**, and every
# test file — including that one — imports from here, so the primitive every gate's error ratio is
# built on has exactly ONE definition. `tests/unit/test_decoder_layer_vs_ref.py`
# (`test_noise_floor_helpers_have_exactly_one_definition`) asserts that by object identity, which is
# the successor to the equality check that guarded the duplication while it existed.


def quantize_like_device(t, dtype):
    """Round `t` to exactly the values the device will hold, via ttnn, and return fp32.

    Host-only (no `device=` argument), so this is a pure quantiser and never a compute path.
    Reproduces `bfloat8_b`'s shared-exponent tile blocking exactly — which no hand-rolled torch
    emulation does. Requires a 4D, tile-shaped tensor for TILE_LAYOUT.

    This is the primitive `DEC-032`'s noise-floor method is built on: quantise every tensor the
    device *stores*, do all the arithmetic in fp32, and PCC that against the fp32 reference. The
    result is implementation-independent and distribution-stable, unlike a PCC copied from another
    implementation whose reference shares the device's own rounding (`BRINGUP_RECIPE.md` E.1).
    """
    import ttnn

    assert t.dim() == 4, f"quantize_like_device expects a 4D tensor, got {tuple(t.shape)}"
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


def err_ratio(measured: float, floor: float) -> float:
    """`(1 - measured) / (1 - floor)` — the measured error in units of the noise floor's.

    `1.0` means the module is exactly at the floor. `20x+` off the floor is a finding even when the
    absolute PCC looks pretty (`BRINGUP_RECIPE.md` E.2) — and `E.5` is the standing caveat: a
    storage-dtype floor does not model a fused kernel's interior, so a large ratio must be
    attributed to a named stage before it is treated as a bug.
    """
    return float("inf") if floor >= 1.0 else (1.0 - float(measured)) / (1.0 - float(floor))
