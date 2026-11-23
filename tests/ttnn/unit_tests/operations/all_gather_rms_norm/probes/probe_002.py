import torch, ttnn
from eval.golden_tests.all_gather_rms_norm.helpers import _pcc_rel_rms, pytorch_all_gather_rms_norm
from tests.ttnn.unit_tests.operations.all_gather_rms_norm.test_all_gather_rms_norm import (
    _make_inputs,
    _make_weight,
    create_ttnn_input_tensor,
    create_ttnn_gamma,
    gathered_shape,
)
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
try:
    for shape, dt, off, topo, links in [
        ((1, 1, 640, 32), ttnn.bfloat16, 0.0, ttnn.Topology.Ring, 2),
        ((1, 1, 128, 384), ttnn.float32, 50.0, ttnn.Topology.Linear, 1),
        ((1, 1, 640, 32), ttnn.float32, 50.0, ttnn.Topology.Ring, 2),
    ]:
        x = _make_inputs(md, shape, dt, layer_norm=True, offset=off)
        W = gathered_shape(shape, 4)[-1]
        g = _make_weight(W, dt, 43)
        b = _make_weight(W, dt, 44)
        tx = create_ttnn_input_tensor(x, md, dtype=dt, layout=ttnn.TILE_LAYOUT)
        tg = create_ttnn_gamma(g, md, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT)
        tb = create_ttnn_gamma(b, md, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT)
        out = all_gather_rms_norm(
            tx, cluster_axis=1, epsilon=1e-6, topology=topo, num_links=links, gamma=tg, beta=tb, norm_type="layer_norm"
        )
        exp = pytorch_all_gather_rms_norm(x, g, 1, 1e-6, norm_type="layer_norm", beta=b)
        outs = ttnn.get_device_tensors(out)
        worst = None
        for c, o in enumerate(outs):
            a = ttnn.to_torch(o).float().reshape(exp[0, c].shape)
            pcc, rms = _pcc_rel_rms(a, exp[0, c].float())
            worst = (pcc, rms) if worst is None or pcc < worst[0] else worst
        print("ACC", shape, dt, off, "pcc=%.6f rms=%.5f" % worst)
finally:
    ttnn.close_mesh_device(md)
