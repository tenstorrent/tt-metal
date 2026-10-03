# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the InstanceNorm / Patch input prep in tt/model_preprocessing."""

from __future__ import annotations

from pathlib import Path

import torch

from models.experimental.chronos_forecast.tt.model_preprocessing import (
    instance_norm,
    instance_norm_inverse,
    patch,
    prepare_patched_context,
    prepare_patched_future,
)

PREPROCESS_SRC = Path(__file__).resolve().parents[1] / "tt" / "model_preprocessing.py"


def test_implementation_does_not_import_chronos():
    text = PREPROCESS_SRC.read_text()
    assert "import chronos" not in text
    assert "from chronos" not in text


def test_instance_norm_matches_standardization_formula():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
    normalized, (loc, scale) = instance_norm(x)
    torch.testing.assert_close(normalized[0], normalized[1])
    torch.testing.assert_close(loc.squeeze(), torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(scale.squeeze(), torch.tensor([1.41421, 1.41421]), atol=1e-4, rtol=1e-4)


def test_instance_norm_preserves_nans_and_inverse():
    x = torch.tensor([[1.0, float("nan"), 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, float("nan")]])
    normalized, loc_scale = instance_norm(x)
    assert torch.equal(normalized.isnan(), x.isnan())
    restored = instance_norm_inverse(normalized, loc_scale)
    torch.testing.assert_close(restored, x, equal_nan=True)


def test_instance_norm_zero_variance_uses_eps():
    x = torch.ones(1, 4)
    _, (loc, scale) = instance_norm(x, eps=1e-5)
    torch.testing.assert_close(loc, torch.ones(1, 1))
    torch.testing.assert_close(scale, torch.full((1, 1), 1e-5))


def test_instance_norm_all_nan_row():
    x = torch.full((1, 3), float("nan"))
    y, (loc, scale) = instance_norm(x)
    torch.testing.assert_close(loc, torch.zeros(1, 1))
    torch.testing.assert_close(scale, torch.ones(1, 1))
    assert torch.isnan(y).all()


def test_instance_norm_arcsinh_and_inverse():
    torch.manual_seed(0)
    x = torch.randn(2, 8)
    y, loc_scale = instance_norm(x, use_arcsinh=True)
    y_no, _ = instance_norm(x, use_arcsinh=False)
    assert not torch.allclose(y, y_no)
    restored = instance_norm_inverse(y, loc_scale, use_arcsinh=True)
    torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-5)


def test_instance_norm_oracle_vs_amazon():
    from models.experimental.chronos_forecast.common.chronos_src import ensure_chronos_on_path

    ensure_chronos_on_path()
    from chronos.chronos_bolt import InstanceNorm as UpInstanceNorm

    torch.manual_seed(0)
    x = torch.randn(3, 16)
    x[0, 3] = float("nan")
    for use_arcsinh in (False, True):
        up = UpInstanceNorm(use_arcsinh=use_arcsinh)
        ref_y, ref_ls = up(x)
        y, ls = instance_norm(x, use_arcsinh=use_arcsinh)
        torch.testing.assert_close(y, ref_y, equal_nan=True, atol=0, rtol=0)
        torch.testing.assert_close(ls[0], ref_ls[0], atol=0, rtol=0)
        torch.testing.assert_close(ls[1], ref_ls[1], atol=0, rtol=0)
        torch.testing.assert_close(
            instance_norm_inverse(y, ls, use_arcsinh=use_arcsinh),
            up.inverse(ref_y, ref_ls),
            equal_nan=True,
            atol=0,
            rtol=0,
        )


def test_patch_matches_vendored_reference():
    from models.experimental.chronos_forecast.reference.chronos_bolt_ops import Patch as RefPatch

    torch.manual_seed(0)
    x = torch.randn(2, 20)
    ref = RefPatch(patch_size=16, patch_stride=16)
    torch.testing.assert_close(patch(x, patch_size=16, patch_stride=16), ref(x), equal_nan=True, atol=0, rtol=0)


def test_patch_oracle_vs_amazon_submodule():
    from models.experimental.chronos_forecast.common.chronos_src import ensure_chronos_on_path

    ensure_chronos_on_path()
    from chronos.chronos_bolt import Patch as UpPatch

    torch.manual_seed(0)
    x = torch.randn(2, 20)
    up = UpPatch(patch_size=16, patch_stride=16)
    torch.testing.assert_close(patch(x, 16, 16), up(x), equal_nan=True, atol=0, rtol=0)


def test_prepare_patched_context_matches_reference_model():
    from models.experimental.chronos_forecast.common.chronos_src import CHRONOS_SUBMODULE_ROOT
    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel

    dummy = CHRONOS_SUBMODULE_ROOT / "test" / "dummy-chronos2-model"
    model = RefModel.from_pretrained(dummy).eval()
    torch.manual_seed(0)
    context = torch.randn(2, 32)
    ref_patched, ref_mask, ref_ls = model._prepare_patched_context(context)
    patched, mask, loc_scale = prepare_patched_context(
        context,
        patch_size=model.chronos_config.input_patch_size,
        patch_stride=model.chronos_config.input_patch_stride,
        context_length=model.chronos_config.context_length,
        time_encoding_scale=model.chronos_config.time_encoding_scale,
        use_arcsinh=model.chronos_config.use_arcsinh,
    )
    torch.testing.assert_close(patched, ref_patched, equal_nan=True, atol=0, rtol=0)
    assert torch.equal(mask, ref_mask)
    torch.testing.assert_close(loc_scale[0], ref_ls[0], atol=0, rtol=0)
    torch.testing.assert_close(loc_scale[1], ref_ls[1], atol=0, rtol=0)


def test_prepare_patched_future_matches_reference_model():
    from models.experimental.chronos_forecast.common.chronos_src import CHRONOS_SUBMODULE_ROOT
    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel

    dummy = CHRONOS_SUBMODULE_ROOT / "test" / "dummy-chronos2-model"
    model = RefModel.from_pretrained(dummy).eval()
    torch.manual_seed(0)
    context = torch.randn(2, 32)
    _, _, loc_scale = model._prepare_patched_context(context)
    future = torch.randn(2, 16)
    future[0, 3] = float("nan")
    ref_patched, ref_mask = model._prepare_patched_future(
        future_covariates=future,
        future_covariates_mask=None,
        loc_scale=loc_scale,
        num_output_patches=1,
        batch_size=2,
    )
    patched, mask = prepare_patched_future(
        future,
        loc_scale,
        num_output_patches=1,
        output_patch_size=model.chronos_config.output_patch_size,
        batch_size=2,
        time_encoding_scale=int(model.chronos_config.time_encoding_scale),
        use_arcsinh=model.chronos_config.use_arcsinh,
    )
    torch.testing.assert_close(patched, ref_patched, equal_nan=True, atol=0, rtol=0)
    torch.testing.assert_close(mask, ref_mask, equal_nan=True, atol=0, rtol=0)
