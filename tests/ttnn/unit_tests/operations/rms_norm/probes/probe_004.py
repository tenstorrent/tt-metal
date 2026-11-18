import sys, itertools, collections
sys.path.insert(0, ".")
import torch, ttnn
from eval.golden_tests.rms_norm.feature_spec import INPUTS, INVALID, TARGET
from eval.golden_tests.rms_norm.helpers import _build_memory_config, _TORCH_DTYPE
from ttnn.operations.rms_norm.rms_norm import INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate
from ttnn.operations.rms_norm.rms_norm_program_descriptor import create_program_descriptor
device = ttnn.open_device(device_id=0)
ML = ttnn.TensorMemoryLayout
fails = collections.Counter(); ok = 0; infeas = 0; examples = {}
gammas = [("no_gamma","none","none")] + [("gamma", d, l) for d in (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b) for l in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT)]
for (shape,) in INPUTS:
  for dtype in TARGET["dtype"]:
    for layout in TARGET["layout"]:
      for ml in (ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED, ML.BLOCK_SHARDED):
        for acc in (True, False):
          for gm, gd, gl in gammas:
            axes = dict(dtype=dtype, layout=layout, memory_layout=ml, fp32_dest_acc_en=acc, gamma_mode=gm, gamma_dtype=gd, gamma_layout=gl)
            for a, t in INPUT_TAGGERS.items(): axes[a] = t((shape,), axes)
            if any(all(axes.get(k) == v for k, v in inv.items()) for inv in INVALID): continue
            if any(all(axes.get(k) == v for k, v in e.items()) for e in EXCLUSIONS): continue
            try:
                mc = _build_memory_config(ml, layout, list(shape), device, dtype=dtype)
                x = ttnn.from_torch(torch.zeros(shape).to(_TORCH_DTYPE[dtype]), dtype=dtype, layout=layout, device=device, memory_config=mc)
                y = ttnn.allocate_tensor_on_device(x.shape, x.dtype, x.layout, device, x.memory_config())
            except Exception as e:
                infeas += 1; continue
            g = None
            if gm == "gamma":
                g = ttnn.from_torch(torch.zeros(1,1,1,shape[-1]).to(_TORCH_DTYPE[gd]), dtype=gd, layout=gl, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=acc, math_approx_mode=False)
            try:
                validate(x, gamma=g, compute_kernel_config=cfg, memory_config=x.memory_config())
                d = create_program_descriptor(x, g, y, 1e-6, cfg)
                ok += 1
            except Exception as e:
                key = (str(ml).split(".")[-1], str(layout).split(".")[-1], type(e).__name__, str(e)[:60])
                fails[key] += 1; examples.setdefault(key, []).append((shape, str(dtype), gd if gd=="none" else str(gd), str(gl), acc))
            del x, y
print("ok", ok, "infeasible(input/output alloc)", infeas)
for k, v in fails.most_common(): print(v, k, examples[k][:4])
ttnn.close_device(device)
