# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Precision analysis step 1: the bfp4 routed-expert weights each MoE path actually holds on the device, decoded
(read back from the device), against host quantizations of the checkpoint weights (fp8 dequantized to fp32, W^T):

    A = bfp4(fp32 W^T)          ttnn.from_torch(fp32, bfloat4_b) on the host
    B = bfp4(bf16(fp32 W^T))    the same after rounding the fp32 weights to bf16

unified (TtRoutedExpert: gate_projs / up_projs / down_projs, chip 0's local slots) and ag (FlatRoutedExpert: chip 0's
w_gu / w_d bank regions, rebuilt from A / B with flat_weight_regions in the op's own layout). Prints, per path and
projection, the fraction of elements equal to A and to B and the max |difference|.
GLM_BITS_LAYER (default 4), GLM_BITS_EXPERTS (unified: local slots checked, default 4)."""

import os

import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
LAYER = int(os.environ.get("GLM_BITS_LAYER", "4"))
N_UNI = int(os.environ.get("GLM_BITS_EXPERTS", "4"))


def _cmp(tag, got, a, b):
    got, a, b = got.float(), a.float(), b.float()
    ea, eb = (got == a).float().mean().item(), (got == b).float().mean().item()
    ab = (a == b).float().mean().item()
    print(
        f"[bits] {tag}: equal to bfp4(fp32) {ea:.6f}  to bfp4(bf16) {eb:.6f}  (A == B {ab:.6f})  "
        f"max|got-A| {(got - a).abs().max().item():.3e}  max|got-B| {(got - b).abs().max().item():.3e}",
        flush=True,
    )


@mesh_parametrize
def test_weight_bits(mesh_device):
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import flat_weight_regions

    import ttnn
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
    from models.demos.glm53_flash_d_p.bringup import hooks
    from models.demos.glm53_flash_d_p.reference.weights import PackedExpert

    hooks.apply_device_settings(S)
    loader, cfg = hooks._loader_cfg(S)
    wdt = hooks.experts_dtype(S)
    assert wdt == ttnn.bfloat4_b
    q4 = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa
    rows, cols = tuple(mesh_device.shape)
    dev0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()  # noqa: E731
    cache = {}

    def wts(g):  # (Wg^T, Wu^T, Wd^T) fp32 exact, A, B
        if g not in cache:
            gw, uw, dw = PackedExpert(loader, LAYER, g).weights(torch.float32)
            ex = [w.T.contiguous() for w in (gw, uw, dw)]
            cache[g] = (ex, [q4(w) for w in ex], [q4(w.bfloat16().float()) for w in ex])
        return cache[g]

    # unified: TtRoutedExpert's per-local-slot tensors, chip 0 = mesh (0, 0)
    os.environ["GLM_EXPERTS_MODE"] = "unified"
    from models.demos.glm53_flash_d_p.tt.experts import build_experts

    uni = build_experts(mesh_device, loader, cfg, LAYER, max(hooks._chunks(S)), weights_dtype=wdt)
    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=uni.epc, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    for le in range(N_UNI):
        g = int(table[0, 0, le])
        _, A, B = wts(g)
        for k, (name, t) in enumerate(
            (("gate", uni.routed.gate_projs[le]), ("up", uni.routed.up_projs[le]), ("down", uni.routed.down_projs[le]))
        ):
            got = dev0(t).reshape(A[k].shape)
            _cmp(f"unified L{LAYER} expert {g} {name}", got, A[k], B[k])
    del uni

    # ag: the flat op's bank regions on chip 0 (global experts gids[0])
    from models.demos.glm53_flash_d_p.tt.experts_ag import build_experts_ag

    ag = build_experts_ag(mesh_device, loader, cfg, LAYER, max(hooks._chunks(S)), weights_dtype=wdt)
    lay = dict(ag.flat.plan)
    banks = lay["banks"]

    def host_layout(regions):  # _bank_sharded's host tensor for one device
        per = -(-len(regions) // banks)
        regions = list(regions) + [torch.zeros_like(regions[0])] * (per * banks - len(regions))
        return torch.cat(
            [torch.cat([regions[b + h * banks] for h in range(per)]).reshape(-1, 32) for b in range(banks)], dim=1
        )

    gids0 = ag.gids[0]
    A_l = [tuple(wts(g)[1]) for g in gids0]
    B_l = [tuple(wts(g)[2]) for g in gids0]
    rA, rB = flat_weight_regions(A_l, lay), flat_weight_regions(B_l, lay)
    names = ("w_gu", "w_d", "w_rd") if lay["rdown"] else ("w_gu", "w_d")
    tensors = {"w_gu": ag.flat.w_gu, "w_d": ag.flat.w_d, "w_rd": ag.flat.w_rd}
    for i, n in enumerate(names):
        got = dev0(tensors[n])
        a, b = host_layout(rA[i]), host_layout(rB[i])
        _cmp(f"ag L{LAYER} chip 0 {n} ({len(rA[i])} regions)", got.reshape(a.shape), a, b)
