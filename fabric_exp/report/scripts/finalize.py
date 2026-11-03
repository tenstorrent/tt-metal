#!/usr/bin/env python3
"""Merge and check the bundle: bandwidth.json = bandwidth_p150.json + bandwidth_galaxy.json; every data/*.json is
checked for the required fields of data/SCHEMA.md and unique ids; prints a per-file summary. Reads and writes only
report/data/."""
import collections, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(HERE), "data")
REQ = ["id", "system", "topology", "config", "test", "send_type", "packet_size_B", "router", "metric", "value", "unit",
       "bound_side", "bound_status", "source_file", "source_section", "job", "date", "same_job_with_main", "pair_key",
       "notes"]
FILES = ["bandwidth", "wire", "slack", "latency", "ccl", "size", "slots", "stability", "fanin", "gaps"]


def main():
    bw = []
    for part in ("bandwidth_p150.json", "bandwidth_galaxy.json"):
        p = os.path.join(D, part)
        if os.path.exists(p):
            bw += json.load(open(p))
    json.dump(bw, open(os.path.join(D, "bandwidth.json"), "w"), indent=1)
    for name in FILES + ["galaxy_v4_only"]:
        p = os.path.join(D, name + ".json")
        if not os.path.exists(p):
            print(f"{name:16} MISSING")
            continue
        d = json.load(open(p))
        missing = collections.Counter(f for r in d for f in REQ if f not in r)
        ids = collections.Counter(r.get("id") for r in d)
        dup = [i for i, n in ids.items() if n > 1]
        bad_same = sum(1 for r in d if not isinstance(r.get("same_job_with_main"), bool))
        routers = collections.Counter(r.get("router") for r in d)
        same = sum(1 for r in d if r.get("same_job_with_main") is True and r.get("router") != "main")
        print(f"{name:16} {len(d):5} records; missing fields {dict(missing) or 0}; duplicate ids {len(dup)}; "
              f"non-bool same_job {bad_same}; v4 records same-job with main {same}; routers {dict(routers.most_common(6))}")


if __name__ == "__main__":
    main()
