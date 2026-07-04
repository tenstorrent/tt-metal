"""Root-cause repro: does an intervening different-length prefill corrupt decoder state?
Deterministic (greedy) so identical inputs MUST give identical outputs unless state leaks.

Run (server stopped):
  docker run --rm --device /dev/tenstorrent/3 -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v /home/ttuser/ttwork/tt-metal:/work -v /home/ttuser/ttwork/qwen3_asr_text_decoder:/models/qwen3_asr_text_decoder \
    -v /home/ttuser/ttwork/qwen3_asr_golden:/golden -v /home/ttuser/.cache/huggingface:/root/.cache/huggingface \
    -e TT_MESH_GRAPH_DESC_PATH=/work/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    -e HF_MODEL=/models/qwen3_asr_text_decoder -e TT_LOGGER_LEVEL=ERROR -e ARCH_NAME=blackhole -e TT_METAL_HOME=/work \
    --cap-add ALL qwen3-asr-server:latest bash -lc \
    'source /opt/venv/bin/activate && cd /work && python3 models/demos/audio/qwen3_asr/tests/repro_state.py'
"""
import os, sys
import numpy as np, torch, ttnn
from models.tt_transformers.tt.model_config import ModelArgs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tt"))
from qwen3_asr_decoder import Qwen3ASRDecoder  # noqa


def gen(model, ie, n=16):
    return model.generate(ie, max_new_tokens=n)


def main():
    dev = ttnn.open_device(device_id=0, trace_region_size=200000000, l1_small_size=65536)
    try:
        args = ModelArgs(dev, max_batch_size=1, max_seq_len=2048)
        sd = args.load_state_dict()
        model = Qwen3ASRDecoder(args, ttnn.bfloat16, dev, sd,
                                args.weight_cache_path(ttnn.bfloat16), use_paged_kv_cache=False)
        X = torch.from_numpy(np.load("/golden/inputs_embeds.npy")).float().unsqueeze(0)  # (1,174,2048)
        Xs = X[:, :100, :].contiguous()                                                  # short (100)
        Xl = torch.cat([X, X[:, 20:, :]], dim=1).contiguous()                            # long (~328)
        print(f"X={X.shape[1]} Xs={Xs.shape[1]} Xl={Xl.shape[1]}")

        base = gen(model, X)
        print("X  (baseline)      :", base[:10])
        r2 = gen(model, X)
        print("X  again (no gap)  :", r2[:10], "  same:", r2 == base)
        _ = gen(model, Xs); print("Xs (short) done")
        r3 = gen(model, X)
        print("X  after Xs        :", r3[:10], "  same:", r3 == base)
        _ = gen(model, Xl); print("Xl (long) done, out_len:", len(_))
        r4 = gen(model, X)
        print("X  after Xl        :", r4[:10], "  same:", r4 == base)
        # repeat the alternation a few times to catch cumulative drift
        ok = True
        for i in range(4):
            gen(model, Xs); gen(model, Xl)
            ri = gen(model, X)
            same = ri == base
            ok = ok and same
            print(f"X after Xs,Xl round {i+1}: same={same}  {'' if same else ri[:10]}")
        print("RESULT:", "STABLE (no state leak)" if (r2 == base and r3 == base and r4 == base and ok) else "STATE LEAK")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
