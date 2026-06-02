import argparse
import ttnn
from models.experimental.opt_transfer.graph import build_graph, RealImpl
from models.experimental.opt_transfer.matcher import LLMClient
from models.experimental.opt_transfer.kb.store import KBStore
from models.experimental.opt_transfer.config import CONFIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="seamless_m4t_v2")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    client = LLMClient()
    kb = KBStore(CONFIG.kb_dir).load()
    device = ttnn.open_device(device_id=0)
    try:
        impl = RealImpl(args.model, device, client, kb)
        graph = build_graph(impl)
        out = graph.invoke({"model": args.model, "iteration": 0, "run_dir": str(CONFIG.run_dir / args.model)})
        print("STATUS:", out["status"], "PCC:", out.get("full_pcc"))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
