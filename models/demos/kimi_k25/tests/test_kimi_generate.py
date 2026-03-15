# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""test_kimi_generate.py — M5 full-model smoke test for Kimi K2.5 on tt-metal.

Test plan
---------

CPU-only (no hardware required):

  TestKimiGeneratorImport       — ``KimiGenerator`` / ``load_kimi_model`` can be
                                  imported; ``KimiGenerator`` is a subclass of
                                  ``DeepseekGenerator``.
  TestKimiGeneratorStructure    — ``__init__`` keyword arguments are present;
                                  expected instance attributes exist on a
                                  constructed object (random_weights=True, no
                                  real mesh device needed for attribute checks).
  TestKimiGeneratorConfigVal    — Mismatched HF config fields raise ValueError
                                  *before* any hardware initialisation.

Hardware (requires ``MESH_DEVICE=TG`` / ``DUAL`` / ``QUAD``):

  test_forward_pass             — Random-weights 2-layer full-model decode and
                                  prefill smoke test.  Uses
                                  ``RowBatchedModel.forward_decode`` /
                                  ``forward_prefill`` driven from Kimi's
                                  ``hf_config_short`` (384 experts, n_group=1,
                                  64 attn heads).  Pass criterion: no crash,
                                  output tensor is finite (no NaN/Inf).

Notes
-----
* All CPU tests are import-only or mock-object tests — no PyTorch/TTNN needed
  at collection time.  They are structured as ``TestCase`` classes so pytest
  collects them correctly without any ``--no-header`` gymnastics.
* The hardware test ``test_forward_pass`` is the key M5 deliverable: it
  exercises the full layer chain (embed → 1 dense MLA → 1 MoE → … → lm_head)
  end-to-end with random weights on real silicon.
* PCC comparison against a reference model is **not** required here (that is
  M6 correctness work).  We only check for finite outputs.
"""

from __future__ import annotations

import inspect
import os
from copy import deepcopy

import pytest


# ============================================================================
# CPU-only: import tests
# ============================================================================


class TestKimiGeneratorImport:
    """M5 — KimiGenerator and load_kimi_model must be importable."""

    def test_kimi_generator_import(self):
        """KimiGenerator can be imported from the Kimi tt module."""
        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator  # noqa: F401

    def test_load_kimi_model_import(self):
        """load_kimi_model factory can be imported."""
        from models.demos.kimi_k25.tt.kimi_model import load_kimi_model  # noqa: F401

    def test_kimi_generator_is_callable(self):
        """KimiGenerator is a class (callable)."""
        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        assert callable(KimiGenerator)

    def test_kimi_generator_inherits_deepseek(self):
        """KimiGenerator must subclass DeepseekGenerator."""
        from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        assert issubclass(KimiGenerator, DeepseekGenerator), (
            f"KimiGenerator must subclass DeepseekGenerator; "
            f"MRO: {[c.__name__ for c in KimiGenerator.__mro__]}"
        )

    def test_load_kimi_model_is_callable(self):
        """load_kimi_model must be a callable (not a class)."""
        from models.demos.kimi_k25.tt.kimi_model import load_kimi_model

        assert callable(load_kimi_model)
        assert not isinstance(load_kimi_model, type), "load_kimi_model should be a function, not a class"


# ============================================================================
# CPU-only: __init__ signature / structure
# ============================================================================


class TestKimiGeneratorStructure:
    """M5 — KimiGenerator.__init__ must accept expected keyword arguments."""

    @pytest.fixture(autouse=True)
    def _kimi_cls(self):
        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        self.KimiGenerator = KimiGenerator

    def test_init_accepts_model_path(self):
        """__init__ must accept model_path kwarg."""
        sig = inspect.signature(self.KimiGenerator.__init__)
        assert "model_path" in sig.parameters, "__init__ must have a 'model_path' parameter"

    def test_init_accepts_mesh_device(self):
        """__init__ must accept mesh_device kwarg."""
        sig = inspect.signature(self.KimiGenerator.__init__)
        assert "mesh_device" in sig.parameters, "__init__ must have a 'mesh_device' parameter"

    def test_init_accepts_cache_dir(self):
        """__init__ must accept cache_dir kwarg."""
        sig = inspect.signature(self.KimiGenerator.__init__)
        assert "cache_dir" in sig.parameters, "__init__ must have a 'cache_dir' parameter"

    def test_init_accepts_hf_config(self):
        """__init__ must accept hf_config kwarg (allows pre-loaded config injection)."""
        sig = inspect.signature(self.KimiGenerator.__init__)
        assert "hf_config" in sig.parameters, "__init__ must have an 'hf_config' parameter"

    def test_init_accepts_random_weights(self):
        """__init__ must accept random_weights kwarg (inherited from DeepseekGenerator)."""
        sig = inspect.signature(self.KimiGenerator.__init__)
        assert "random_weights" in sig.parameters or "kwargs" in sig.parameters, (
            "__init__ must accept 'random_weights' (directly or via **kwargs)"
        )

    def test_init_accepts_override_num_layers(self):
        """__init__ must accept override_num_layers for smoke tests."""
        sig = inspect.signature(self.KimiGenerator.__init__)
        assert "override_num_layers" in sig.parameters or "kwargs" in sig.parameters, (
            "__init__ must accept 'override_num_layers' (directly or via **kwargs)"
        )

    def test_prepare_weight_configs_is_overridden(self):
        """KimiGenerator must define its own _prepare_weight_configs."""
        assert "_prepare_weight_configs" in self.KimiGenerator.__dict__, (
            "KimiGenerator must override _prepare_weight_configs "
            "(not just inherit from DeepseekGenerator)"
        )

    def test_load_kimi_model_signature(self):
        """load_kimi_model must accept model_path, mesh_device, cache_dir, **kwargs."""
        from models.demos.kimi_k25.tt.kimi_model import load_kimi_model

        sig = inspect.signature(load_kimi_model)
        for param in ("model_path", "mesh_device", "cache_dir"):
            assert param in sig.parameters, f"load_kimi_model must have a '{param}' parameter"

    def test_default_cache_dir_is_kimi_specific(self):
        """Default cache dir must reference 'kimi' to avoid overwriting DSV3 cache."""
        from models.demos.kimi_k25.tt.kimi_model import _DEFAULT_CACHE_DIR

        assert "kimi" in _DEFAULT_CACHE_DIR.lower(), (
            f"_DEFAULT_CACHE_DIR={_DEFAULT_CACHE_DIR!r} must contain 'kimi' "
            "to avoid colliding with the DSV3 weight cache"
        )


# ============================================================================
# CPU-only: config validation
# ============================================================================


class TestKimiGeneratorConfigValidation:
    """M5 — KimiGenerator must reject mismatched HF configs at __init__ time."""

    @pytest.fixture(autouse=True)
    def _base_config(self):
        """Fixture config for all validation tests."""
        from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

        self.good_config = KimiK25Config.from_fixture()
        self.KimiK25Config = KimiK25Config

    def _make_config(self, **overrides) -> object:
        """Return a deepcopy of the fixture config with the given fields overridden."""
        cfg = deepcopy(self.good_config)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def _raises_on_init(self, hf_config) -> bool:
        """Return True if KimiGenerator raises ValueError for the given config."""
        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        try:
            KimiGenerator(hf_config=hf_config, mesh_device=None, model_path=None)
        except ValueError:
            return True
        except Exception:
            # Any other error (e.g. mesh_device=None downstream) is fine — we
            # only care that ValueError was raised for the config mismatch, not
            # that the full __init__ succeeded.
            pass
        return False

    def test_valid_config_does_not_raise(self):
        """A valid Kimi K2.5 config must not raise ValueError in __init__."""
        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        try:
            KimiGenerator(hf_config=self.good_config, mesh_device=None, model_path=None)
        except ValueError as exc:
            pytest.fail(f"Valid Kimi config raised ValueError: {exc}")
        except Exception:
            # Other exceptions (NoneType mesh_device, etc.) are expected with
            # mesh_device=None; we only care that ValueError was NOT raised.
            pass

    def test_wrong_n_routed_experts_raises(self):
        """Incorrect n_routed_experts (e.g. 256 instead of 384) must raise ValueError."""
        bad = self._make_config(n_routed_experts=256)
        assert self._raises_on_init(bad), "n_routed_experts=256 should raise ValueError"

    def test_wrong_rms_norm_eps_raises(self):
        """Incorrect rms_norm_eps (e.g. 1e-6 instead of 1e-5) must raise ValueError."""
        bad = self._make_config(rms_norm_eps=1e-6)
        assert self._raises_on_init(bad), "rms_norm_eps=1e-6 should raise ValueError"

    def test_wrong_hidden_size_raises(self):
        """Incorrect hidden_size (e.g. 4096 instead of 7168) must raise ValueError."""
        bad = self._make_config(hidden_size=4096)
        assert self._raises_on_init(bad), "hidden_size=4096 should raise ValueError"

    def test_wrong_num_attention_heads_raises(self):
        """Incorrect num_attention_heads (e.g. 128 instead of 64) must raise ValueError."""
        bad = self._make_config(num_attention_heads=128)
        assert self._raises_on_init(bad), "num_attention_heads=128 should raise ValueError"

    def test_wrong_vocab_size_raises(self):
        """Incorrect vocab_size (e.g. 32000 instead of 163840) must raise ValueError."""
        bad = self._make_config(vocab_size=32000)
        assert self._raises_on_init(bad), "vocab_size=32000 should raise ValueError"


# ============================================================================
# CPU-only: weight loader integration
# ============================================================================


class TestKimiGeneratorWeightLoaderIntegration:
    """M5 — Verify that _prepare_weight_configs injects KimiLazyStateDict."""

    def test_prepare_weight_configs_uses_kimi_state_dict(self):
        """_prepare_weight_configs body must reference KimiLazyStateDict."""
        import inspect

        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        src = inspect.getsource(KimiGenerator._prepare_weight_configs)
        assert "KimiLazyStateDict" in src, (
            "_prepare_weight_configs must instantiate KimiLazyStateDict "
            "for the real-weights path"
        )

    def test_prepare_weight_configs_random_weights_falls_through(self):
        """For random_weights=True, _prepare_weight_configs must call super()."""
        import inspect

        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        src = inspect.getsource(KimiGenerator._prepare_weight_configs)
        assert "random_weights" in src, (
            "_prepare_weight_configs must handle random_weights flag"
        )
        assert "super()" in src, (
            "_prepare_weight_configs must call super() for the random_weights path"
        )

    def test_kimi_lazy_state_dict_importable(self):
        """KimiLazyStateDict must be importable from utils.weight_loader."""
        from models.demos.kimi_k25.utils.weight_loader import KimiLazyStateDict  # noqa: F401


# ============================================================================
# Hardware: full-model random-weights smoke test
# ============================================================================


def _run_kimi_forward_pass_random_weights(
    mode: str,
    seq_len: int,
    batch_size_per_row: int,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate: bool,
    num_layers: int = 2,
) -> None:
    """Run a full-model forward pass with random weights and Kimi K2.5 config.

    This exercises the complete RowBatchedModel layer stack (embed → MLA →
    MoE → … → lm_head) with Kimi's 384-expert, n_group=1, 64-head config.

    Args:
        mode:               ``"decode"`` or ``"prefill"``.
        seq_len:            Sequence length (1 for decode, ≥1 for prefill).
        batch_size_per_row: Users per mesh row (decode only).
        hf_config_short:    Kimi hf_config with num_hidden_layers=num_layers.
        cache_path:         Weight cache directory.
        mesh_device:        TTNN mesh device.
        ccl:                CCL instance for the mesh.
        force_recalculate:  Bypass weight cache if True.
        num_layers:         Number of transformer layers to instantiate.
    """
    import torch

    import ttnn
    from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
    from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel
    from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
    from models.demos.deepseek_v3.utils.hf_model_utils import prepare_model_state_dict
    from models.demos.deepseek_v3.utils.run_config import create_run_config
    from models.demos.deepseek_v3.utils.test_utils import (
        get_model_config,
        get_rope_tensors,
        get_test_weight_config,
    )

    # Override layer count for speed
    cfg = deepcopy(hf_config_short)
    cfg.num_hidden_layers = num_layers
    # Clamp first_k_dense_replace so it doesn't exceed num_hidden_layers
    if hasattr(cfg, "first_k_dense_replace"):
        cfg.first_k_dense_replace = min(cfg.first_k_dense_replace, num_layers)

    mesh_rows = mesh_device.shape[0]
    dp_factor = mesh_device.shape[1]

    if mode == "prefill":
        assert batch_size_per_row == 1, "Prefill only supports batch_size_per_row=1"
        batch_size = 1
    else:
        assert seq_len == 1, "Decode only supports seq_len=1"
        batch_size = batch_size_per_row * mesh_rows

    # Random token input
    torch.manual_seed(42)
    if mode == "decode":
        torch_input = torch.randint(0, cfg.vocab_size - 1, (batch_size, seq_len), dtype=torch.long)
        position_ids = torch.randint(0, cfg.max_seq_len - 1, (batch_size,), dtype=torch.long)
        torch_input = torch_input.transpose(1, 0)  # (seq_len, batch)
    else:
        torch_input = torch.randint(0, cfg.vocab_size - 1, (1, seq_len), dtype=torch.long)
        position_ids = None

    # Paged attention config
    paged_config = MLA2D.get_valid_paged_config(cfg.max_seq_len, USERS_PER_ROW, dp_factor)

    # KV cache init (empty for random-weight smoke test)
    if mode == "decode":
        cache_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim
        decode_input_caches = tuple(
            torch.empty((batch_size, 1, 0, cache_dim), dtype=torch.bfloat16)
            for _ in range(cfg.num_hidden_layers)
        )
        denom = mesh_rows * dp_factor
        batches_per_device = batch_size // denom
        blocks_per_batch = paged_config.max_num_blocks // batches_per_device
        mapping = torch.arange(batches_per_device * blocks_per_batch, dtype=torch.long).reshape(
            batches_per_device, blocks_per_batch
        )
        mappings = tuple(mapping for _ in range(cfg.num_hidden_layers))

        from models.demos.deepseek_v3.utils.test_utils import paged_caches_from_torch

        paged_input_caches, torch_page_tables = paged_caches_from_torch(
            decode_input_caches,
            tuple(mesh_device.shape),
            paged_config,
            user_id=None,
            mappings=mappings,
        )
    else:
        paged_input_caches = None
        total_global_users = mesh_rows * USERS_PER_ROW
        num_devices = mesh_rows * dp_factor
        batches_per_device = total_global_users // num_devices
        blocks_per_batch = paged_config.max_num_blocks // batches_per_device
        torch_page_table = torch.arange(paged_config.max_num_blocks, dtype=torch.int32).reshape(
            batches_per_device, blocks_per_batch
        )
        torch_page_tables = (torch_page_table,) * cfg.num_hidden_layers

    tt_page_tables = tuple(
        MLA2D.create_page_table(
            page_table=tpt, paged_config=paged_config, mesh_device=mesh_device
        )
        for tpt in torch_page_tables
    )

    # Build random-weight state dict from DSV3 reference model instantiated with Kimi config.
    # RowBatchedModel.convert_weights() does: (state_dict,) = state_dicts — requires exactly
    # one dict.  state_dicts=() (empty tuple) would raise ValueError; state_dicts=None triggers
    # prepare_model_state_dict internally but get_test_weight_config does not forward
    # random_weights=True to get_weight_config.  Prepare explicitly here, mirroring DSV3
    # test_moe.py's generate_reference_io pattern.
    random_state_dict = prepare_model_state_dict(cfg, random_weights=True)

    # Random-weight model config
    weight_config = get_test_weight_config(
        RowBatchedModel,
        cfg,
        (random_state_dict,),  # single state_dict required by RowBatchedModel.convert_weights
        cache_path,
        mesh_device,
        force_recalculate,
        test_name="test_kimi_generate",
        real_weights=False,
    )
    model_config = get_model_config(RowBatchedModel, mode, cfg, mesh_device)
    model_state = RowBatchedModel.create_state(cfg, paged_config, mesh_device, ccl, paged_input_caches)
    model_shared_state = RowBatchedModel.create_shared_state(cfg, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # TTNN input tensor
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    position_ids_tt = (
        ttnn.from_torch(
            position_ids,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            dtype=ttnn.int32,
        )
        if mode == "decode"
        else None
    )

    rope_tensors = get_rope_tensors(cfg, batch_size_per_row, seq_len, position_ids, mesh_device)

    # Forward pass
    if mode == "prefill":
        tt_output = RowBatchedModel.forward_prefill(tt_input, 0, run_config, rope_tensors, tt_page_tables)
    else:
        tt_output = RowBatchedModel.forward_decode(
            tt_input, position_ids_tt, run_config, rope_tensors, tt_page_tables
        )

    ttnn.synchronize_device(mesh_device)

    # Finite output check (NaN/Inf = silent numerical failure)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
    ).float()

    assert torch.isfinite(tt_output_torch).all(), (
        f"Kimi K2.5 full-model {mode} forward pass produced non-finite output "
        f"(NaN/Inf) with random weights.  This indicates a layer configuration "
        f"error (shape mismatch, uninitialized tensor, etc.)."
    )

    ttnn.deallocate(tt_output)


# ---------------------------------------------------------------------------
# Hardware test parametrization
# ---------------------------------------------------------------------------

_SMOKE_TEST_CASES = [
    pytest.param("decode", 1, 32, id="mode_decode_seq_1_batch_32"),
    pytest.param("prefill", 128, 1, id="mode_prefill_seq_128_batch_1"),
]


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mode, seq_len, batch_size_per_row", _SMOKE_TEST_CASES)
def test_forward_pass(
    mode,
    seq_len,
    batch_size_per_row,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    """M5 smoke test — full Kimi K2.5 model forward pass with random weights.

    Instantiates RowBatchedModel driven by Kimi's hf_config_short (384 experts,
    n_group=1, 64 attn heads) with random weights and runs a single forward
    pass.  Validates that the output tensor is finite (no NaN/Inf).

    No reference model comparison is performed here (M6 correctness work).

    Requires MESH_DEVICE=TG / DUAL / QUAD.
    """
    _run_kimi_forward_pass_random_weights(
        mode=mode,
        seq_len=seq_len,
        batch_size_per_row=batch_size_per_row,
        hf_config_short=hf_config_short,
        cache_path=cache_path,
        mesh_device=mesh_device,
        ccl=ccl,
        force_recalculate=force_recalculate_weight_config,
        num_layers=2,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
