"""Error map by producing worker: which (source chip, global head, mt) output tile rows are wrong.

Refinement 2 (split read NoC): global head g on a source chip is computed by worker core index g
(row-major over the grid), so the head list says which cores / which read-NoC set produced bad rows.
"""
import pytest, torch, ttnn
from ttnn.operations.a2a_wkv_b2 import a2a_wkv_b2
from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import MESH_SHAPE, _make_inputs, _expected


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_probe_errmap_heads(mesh_device):
    grid_x = mesh_device.compute_with_storage_grid_size().x
    chips, attn, w = _make_inputs(mesh_device, 0)
    exp = _expected(mesh_device, chips, w)
    out = a2a_wkv_b2(attn, w)
    act = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]
    rows, cols = tuple(mesh_device.shape)
    bad_by_src = {}
    for c, (a, e) in enumerate(zip(act, exp)):
        r = c % cols
        err = (a - e).abs()
        bad = err.reshape(24, cols, 5, 32, 128).amax(dim=(3, 4)) > 0.5  # head, src ring pos j, mt
        for h, j, mt in bad.nonzero().tolist():
            src_chip = (c // cols) * cols + j
            g = r * 24 + h
            bad_by_src.setdefault(src_chip, []).append((g, mt))
    for src, lst in sorted(bad_by_src.items()):
        heads = sorted({g for g, _ in lst})
        par = [((g % grid_x) + (g // grid_x)) % 2 for g in heads]
        mts = sorted({mt for _, mt in lst})
        print(f"src chip {src}: {len(lst)} bad rows; heads {heads}; parity {par}; mts {mts}")
    print("total bad", sum(len(v) for v in bad_by_src.values()))


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_probe_badrow_content(mesh_device):
    """For each bad (dest chip, head, src j, mt) row: is it zeros, non-finite, or another row's expected data?"""
    chips, attn, w = _make_inputs(mesh_device, 0)
    exp = _expected(mesh_device, chips, w)
    out = a2a_wkv_b2(attn, w)
    act = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]
    cols = tuple(mesh_device.shape)[1]
    shown = 0
    for c, (a, e) in enumerate(zip(act, exp)):
        A = a.reshape(24, cols, 5, 32, 128)
        E = e.reshape(24, cols, 5, 32, 128)
        bad = (A - E).abs().amax(dim=(3, 4)) > 0.5
        for h, j, mt in bad.nonzero().tolist():
            blk = A[h, j, mt]
            row_bad = ((blk - E[h, j, mt]).abs().amax(dim=1) > 0.5).nonzero().flatten().tolist()
            match = None
            for h2 in range(24):
                for j2 in range(cols):
                    for m2 in range(5):
                        if (blk - E[h2, j2, m2]).abs().max() < 0.5:
                            match = (h2, j2, m2)
            # previous-call candidate: same dest, any row matching partially
            print(
                f"dest {c} h {h} j {j} mt {mt}: bad rows {row_bad[:4]}..({len(row_bad)}) zeros={bool((blk == 0).all())} "
                f"finite={bool(torch.isfinite(blk).all())} absmax={blk.abs().max():.3g} matches={match}"
            )
            shown += 1
            if shown > 30:
                return
