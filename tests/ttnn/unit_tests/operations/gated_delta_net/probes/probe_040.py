# Bit-identity of the role split against HEAD's descriptor + kernels (copied to /tmp/gdn_head), one shape per regime.
import importlib.util, torch, ttnn
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs
import sys, ttnn.operations.gated_delta_net

gm = sys.modules["ttnn.operations.gated_delta_net.gated_delta_net"]
cur_cached = gm.cached_program_descriptor
spec = importlib.util.spec_from_file_location("gdn_head_pd", "/tmp/gdn_head/gated_delta_net_program_descriptor.py")
head_pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(head_pd)
device = ttnn.open_device(device_id=0)
cfg = compute_config()
SHAPES = {
    "decode_b1": ((1, 1, 2560), (1, 12, 128, 128)),
    "decode_b8": ((1, 8, 2560), (8, 12, 128, 128)),
    "decode_b16": ((1, 16, 2560), (16, 12, 128, 128)),
    "prefill_t32": ((1, 32, 2560), (1, 12, 128, 128)),
    "prefill_t96": ((1, 96, 2560), (1, 12, 128, 128)),
    "prefill_t128": ((1, 128, 2560), (1, 12, 128, 128)),
    "batched_b2_t40": ((1, 80, 2560), (2, 12, 128, 128)),
    "batched_b2_t2": ((1, 4, 2560), (2, 12, 128, 128)),
    "small_t5": ((1, 5, 768), (1, 4, 64, 64)),
    "prefill_4096": ((1, 4096, 2560), (1, 12, 128, 128)),
}


def upload(t, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


for name, shape in SHAPES.items():
    for tdt in (ttnn.bfloat16, ttnn.float32):
        torch.manual_seed(0)
        try:
            tensors = make_inputs(shape)
        except Exception as e:
            print(f"BITS {name} skip: {e}")
            break
        results = {}
        for impl, fn in (("cur", cur_cached), ("head", head_pd.cached_program_descriptor)):
            gm.cached_program_descriptor = fn
            dev = {n: upload(tensors[n], ttnn.float32 if n == "recurrent_state" else tdt) for n in ORDER}
            try:
                outs = gm.gated_delta_net(*(dev[n] for n in ORDER), compute_kernel_config=cfg)
                outs_t = [ttnn.to_torch(o) for o in outs]
                cs, rs = dev["conv_state"], dev["recurrent_state"]
                outs_ip = gm.gated_delta_net(
                    *(dev[n] for n in ORDER), compute_kernel_config=cfg, conv_state_out=cs, recurrent_state_out=rs
                )
                outs_t += [ttnn.to_torch(o) for o in outs_ip]
                results[impl] = outs_t
            except Exception as e:
                results[impl] = repr(e)[:120]
        if isinstance(results["cur"], str) or isinstance(results["head"], str):
            print(
                f"BITS {name} {tdt}: cur={results['cur'] if isinstance(results['cur'], str) else 'ok'} head={results['head'] if isinstance(results['head'], str) else 'ok'}"
            )
            continue
        same = [torch.equal(a, b) for a, b in zip(results["cur"], results["head"])]
        print(f"BITS {name} {tdt}: {'IDENTICAL' if all(same) else 'DIFF ' + str(same)}")
gm.cached_program_descriptor = cur_cached
ttnn.close_device(device)
