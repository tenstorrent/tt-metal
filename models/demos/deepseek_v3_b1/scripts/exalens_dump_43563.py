"""Dump TensixState from halted SDPA cores for #43563.

Used together with the HALT_FLASH_MLA_ITER ebreak in flash_mla.hpp:
  TT_HALT_FLASH_MLA_ITER=0 pytest ...   # halt at iter-0 entry
  TT_HALT_FLASH_MLA_ITER=1 pytest ...   # halt at iter-1 entry

Run this script while pytest is hung at the ebreak. Dumps one
representative SDPA worker core's full TensixState to a JSON file.

Two runs (iter-0 then iter-1) produce two dump files; diff them to
identify bank-conditional state.
"""

import argparse
import dataclasses
import json
import sys

from ttexalens.tt_exalens_lib import get_tensix_state, init_ttexalens


def state_to_dict(state):
    """Convert TensixState (or any dataclass-like object) to a serializable dict."""
    if state is None:
        return None
    if isinstance(state, (int, float, str, bool)):
        return state
    if isinstance(state, (list, tuple)):
        return [state_to_dict(v) for v in state]
    if isinstance(state, dict):
        return {str(k): state_to_dict(v) for k, v in state.items()}
    if dataclasses.is_dataclass(state):
        return {f.name: state_to_dict(getattr(state, f.name)) for f in dataclasses.fields(state)}
    # Fallback: try __dict__
    d = getattr(state, "__dict__", None)
    if d is not None:
        return {k: state_to_dict(v) for k, v in d.items() if not k.startswith("_")}
    return repr(state)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=6, help="device id (0..7)")
    p.add_argument("--core", default="0,1", help="logical core x,y (default 0,1 — first SDPA sender on device 6)")
    p.add_argument("--out", required=True, help="output JSON file")
    args = p.parse_args()

    ctx = init_ttexalens(safe_mode=True)
    cx, cy = map(int, args.core.split(","))
    location_str = f"{cx}-{cy}"

    print(f"Attaching to device {args.device}, core {location_str}...", file=sys.stderr)
    state = get_tensix_state(location_str, device_id=args.device, context=ctx)
    out = state_to_dict(state)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=repr)
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
