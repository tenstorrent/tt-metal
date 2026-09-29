import collections, yaml, ttnn

try:
    path = ttnn.cluster.serialize_cluster_descriptor()
    print("DESC", path)
    d = yaml.safe_load(open(path))
    pairs = collections.Counter()
    for a, b in d.get("ethernet_connections", []):
        pairs[tuple(sorted((a["chip"], b["chip"])))] += 1
    for (x, y), n in sorted(pairs.items()):
        print(f"PAIR chip {x} <-> chip {y}: {n} ethernet cables")
    print("TOTAL", sum(pairs.values()), "chip-to-chip connections")
except Exception as e:
    print("ERR", type(e).__name__, e)
