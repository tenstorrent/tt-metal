# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Certify each host fallback against the real ttnn op on the captured qwen3_vl_ops cases (WH/BH only)."""
import importlib
import json

import pytest

import ttnn
from models.common.utility_functions import is_quasar
from models.experimental.ops.quasar.qwen3_vl.tests.e2e import op_overrides as O
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.pcc import pcc
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.recorder import to_host
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G

# Fallback target -> generated qwen3_vl_ops module holding its captured CASES. ttnn.matmul and
# chunked_scaled_dot_product_attention have no captured calls, so they stay uncertified.
_CASE_MODULE = {
    "ttnn.linear": "test_linear",
    "ttnn.rms_norm": "test_rms_norm",
    "ttnn.layer_norm": "test_layer_norm",
    "ttnn.add": "test_add",
    "ttnn.multiply": "test_multiply",
    "ttnn.transformer.scaled_dot_product_attention": "test_scaled_dot_product_attention",
    "ttnn.transformer.paged_scaled_dot_product_attention_decode": "test_paged_scaled_dot_product_attention_decode",
    "ttnn.experimental.minimal_matmul": "test_minimal_matmul",
    "ttnn.experimental.nlp_create_qkv_heads": "test_nlp_create_qkv_heads",
    "ttnn.experimental.nlp_create_qkv_heads_decode": "test_nlp_create_qkv_heads_decode",
    "ttnn.experimental.nlp_concat_heads": "test_nlp_concat_heads",
    "ttnn.experimental.nlp_concat_heads_decode": "test_nlp_concat_heads_decode",
    "ttnn.experimental.rotary_embedding_llama": "test_rotary_embedding_llama",
    "ttnn.experimental.paged_update_cache": "test_paged_update_cache",
    "ttnn.experimental.paged_fill_cache": "test_paged_fill_cache",
}


def _bf16(case):
    """Captured cases may use bfp8/bfp4; certify in bf16 like the Quasar config."""
    text = json.dumps(case)
    return json.loads(text.replace('"BFLOAT8_B"', '"BFLOAT16"').replace('"BFLOAT4_B"', '"BFLOAT16"'))


# Captures drop compute_kernel_config, so these ops would use their defaults (often fp32 dest acc, which ttsim WH
# rejects, S1/S2); run them with the Quasar config's HiFi4 + bf16 dest acc instead.
_TAKES_COMPUTE_CONFIG = {
    "ttnn.linear",
    "ttnn.rms_norm",
    "ttnn.layer_norm",
    "ttnn.transformer.scaled_dot_product_attention",
    "ttnn.transformer.paged_scaled_dot_product_attention_decode",
    "ttnn.experimental.minimal_matmul",
}


def _quasar_compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )


def _params():
    out = []
    for target, mod in _CASE_MODULE.items():
        m = importlib.import_module(f"models.experimental.ops.quasar.tests.qwen3_vl_ops.{mod}")
        out += [pytest.param(target, _bf16(c), id=f"{mod}-{c['id']}") for c in m.CASES]
    return out


def _host(v):
    return to_host(v) if isinstance(v, ttnn.Tensor) else v


@pytest.mark.parametrize("target, case", _params())
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fallback_matches_op(mesh_device, target, case):
    if is_quasar():
        pytest.skip("fallbacks are certified on WH/BH, never on Quasar")
    fb = O.FALLBACKS[target]
    args, kwargs, _ = G.build_inputs_for_case(case, mesh_device)
    if target in _TAKES_COMPUTE_CONFIG:
        kwargs.setdefault("compute_kernel_config", _quasar_compute_config())
    # Host copies before the real op runs, since some ops update an input in place.
    targs, tkw = [_host(a) for a in args], {k: _host(v) for k, v in kwargs.items()}
    real = getattr(*O.resolve(target))(*args, **kwargs)
    mine = fb.torch_fn(targs, tkw)
    if fb.inplace_arg is not None:
        pairs = [(to_host(args[fb.inplace_arg]), targs[fb.inplace_arg])]
    else:
        reals = real if isinstance(real, (list, tuple)) else [real]
        mines = mine if isinstance(mine, (list, tuple)) else [mine]
        assert len(reals) == len(mines), f"{target}: {len(reals)} outputs vs {len(mines)} from the fallback"
        pairs = [(to_host(r), m.float()) for r, m in zip(reals, mines)]
    for r, m in pairs:
        assert r.dim() == m.dim(), f"{target} {case['id']}: rank {tuple(r.shape)} vs {tuple(m.shape)}"
        assert all(a >= b for a, b in zip(r.shape, m.shape)), f"{target}: {tuple(r.shape)} < {tuple(m.shape)}"
        r = r[tuple(slice(0, s) for s in m.shape)]  # drop device-side padding only
        p = pcc(r, m)
        assert p >= 0.999, f"{target} {case['id']}: pcc {p:.5f}"
