# SPDX-License-Identifier: Apache-2.0
"""
Profile a Qwen3.6-27B decode step by layer type (linear_attention/full_attention/mlp)
on one P300 chip. Monkeypatches the decoder layer + sublayers to accumulate
synchronized timings, so we know where the ~170ms/token goes before optimizing.

  python3 profile_decode.py --model-path /home/ttuser/ttwork/qwen36-weights --warmup 3 --steps 5
  python3 profile_decode.py --dummy --layers 8           # quick wiring check
"""
import argparse, time, collections
import torch
import ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import load_state_dict, create_dummy_state_dict
from models.demos.qwen36_27b.tt.model import TtQwen36Model
from models.demos.qwen36_27b.tt.generator import Qwen36Generator
from models.demos.qwen36_27b.tt import decoder as decoder_mod

TIMINGS = collections.defaultdict(float)
COUNTS = collections.defaultdict(int)
_dev = None


def main():
    global _dev
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--model-path", type=str, default="/home/ttuser/ttwork/qwen36-weights")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    config = Qwen36ModelConfig()
    if args.layers is not None:
        config.num_hidden_layers = args.layers

    if args.dummy:
        state_dict = create_dummy_state_dict(config, num_layers=config.num_hidden_layers)
        input_ids = torch.tensor([[1, 42, 100, 7, 88, 9]])
    else:
        state_dict = load_state_dict(config, max_layers=args.layers, model_path=args.model_path)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])

    _dev = ttnn.open_device(device_id=0)

    # --- instrument the decoder layer forward ---
    Layer = decoder_mod.TtHybridDecoderLayer
    orig_forward = Layer.forward

    def timed_forward(self, hidden_states, **kw):
        # time the token mixer (attn/deltanet) and the mlp separately
        ttnn.synchronize_device(_dev)
        t0 = time.perf_counter()
        out = orig_forward(self, hidden_states, **kw)
        ttnn.synchronize_device(_dev)
        dt = time.perf_counter() - t0
        TIMINGS[self.layer_type] += dt
        COUNTS[self.layer_type] += 1
        return out

    Layer.forward = timed_forward

    try:
        model = TtQwen36Model(_dev, state_dict, config)
        gen = Qwen36Generator(model, config)
        del state_dict

        gen.prefill(input_ids)
        next_tok = 5
        # warmup (compiles kernels)
        for _ in range(args.warmup):
            _, nt = gen.decode_one_token(torch.tensor([[next_tok]]))
            next_tok = int(nt.item())

        TIMINGS.clear(); COUNTS.clear()
        step_total = []
        for _ in range(args.steps):
            ttnn.synchronize_device(_dev)
            t0 = time.perf_counter()
            _, nt = gen.decode_one_token(torch.tensor([[next_tok]]))
            ttnn.synchronize_device(_dev)
            step_total.append(time.perf_counter() - t0)
            next_tok = int(nt.item())

        print("\n=== decode profile (%d steps, %d layers) ===" % (args.steps, config.num_hidden_layers))
        tot = sum(step_total) / len(step_total)
        print("avg full step: %.1f ms (%.2f tok/s)" % (tot * 1000, 1 / tot))
        for lt in ("linear_attention", "full_attention"):
            n = COUNTS.get(lt, 0)
            if n:
                per_step = TIMINGS[lt] / args.steps
                per_layer = TIMINGS[lt] / n
                nlayers = n // args.steps
                print("  %-17s %2d layers/step  %6.1f ms/step (%.2f ms/layer)  = %4.1f%%"
                      % (lt, nlayers, per_step * 1000, per_layer * 1000, 100 * per_step / tot))
    finally:
        ttnn.close_device(_dev)


if __name__ == "__main__":
    main()
