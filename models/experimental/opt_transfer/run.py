import argparse
import ttnn
from langgraph.checkpoint.sqlite import SqliteSaver
from models.experimental.opt_transfer.graph import build_graph, RealImpl
from models.experimental.opt_transfer.matcher import LLMClient
from models.experimental.opt_transfer.kb.store import KBStore
from models.experimental.opt_transfer.config import CONFIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="seamless_m4t_v2")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue the saved run for --model from its last checkpoint",
    )
    args = ap.parse_args()

    run_dir = CONFIG.run_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    client = LLMClient()
    kb = KBStore(CONFIG.kb_dir).load()
    device = ttnn.open_device(device_id=0)
    try:
        with SqliteSaver.from_conn_string(str(run_dir / "state.db")) as cp:
            graph = build_graph(RealImpl(args.model, device, client, kb), checkpointer=cp)
            cfg = {"configurable": {"thread_id": args.model}}
            init = (
                None  # None -> resume from checkpoint
                if args.resume
                else {"model": args.model, "iteration": 0, "run_dir": str(run_dir)}
            )
            out = graph.invoke(init, cfg)
            print("STATUS:", out["status"], "PCC:", out.get("full_pcc"))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
