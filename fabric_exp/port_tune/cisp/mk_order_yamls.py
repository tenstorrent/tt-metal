#!/usr/bin/env python3
"""mk_order_yamls.py: the row-order variants of pcs_nx_ci_x8.yaml for the CI streaming-profiler stall hunt (approved
2026-11-05; yaml only). Each keeps 8 repeats, 100k packets, num_links 1 and 2, sizes 2048 and 4096.
  A pcs_nx_wonly_x8.yaml: unicast_write rows only (12 back-to-back write iterations per repeat).
  B pcs_nx_wlast_x8.yaml: the stock rows with ntype order fused_atomic_inc, unicast_scatter_write, unicast_write (the
    write block moves from positions 0-5 / 18-23 to 12-17 / 30-35 of each 36-iteration repeat).
  C pcs_nx_wgap_x8.yaml: unicast_write only, one iteration per row, each followed by a tiny non-write row
    (fused_atomic_inc, 2048 B, 1000 packets, same num_links): writes never run back to back.
Test names keep the "_size_S_ntype_T_num_links_L" suffix so the analysis keys rows by it."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD = """allocation_policies:
  receiver:
    policy: RoundRobin
    max_configs_per_core: 4
Tests:
"""


def param_test(name, ntypes):
    return f"""- name: {name}
  benchmark_mode: true
  sync: true
  fabric_setup:
    topology: NeighborExchange
  parametrization_params:
    num_links:
    - 1
    - 2
    ntype:
""" + "".join(f"    - {t}\n" for t in ntypes) + """    size:
    - 2048
    - 4096
  defaults:
    ftype: unicast
    num_packets: 100000
  patterns:
  - type: neighbor_exchange
    iterations: 3
"""


def one_test(name, ntype, size, links, packets):
    return f"""- name: {name}_size_{size}_ntype_{ntype}_num_links_{links}
  benchmark_mode: true
  sync: true
  fabric_setup:
    topology: NeighborExchange
    num_links: {links}
  defaults:
    ftype: unicast
    ntype: {ntype}
    size: {size}
    num_packets: {packets}
  patterns:
  - type: neighbor_exchange
    iterations: 1
"""


def write(fname, comment, body):
    with open(os.path.join(HERE, fname), "w") as f:
        f.write(comment + HEAD + body)


write("pcs_nx_wonly_x8.yaml", "# cisp row-order A (mk_order_yamls.py): unicast_write rows only, 8 repeats.\n",
      "".join(param_test(f"NeighborExchangeUnicastR{r:02d}", ["unicast_write"]) for r in range(1, 9)))
write("pcs_nx_wlast_x8.yaml", "# cisp row-order B (mk_order_yamls.py): stock rows, unicast_write block last, 8 repeats.\n",
      "".join(param_test(f"NeighborExchangeUnicastR{r:02d}", ["fused_atomic_inc", "unicast_scatter_write", "unicast_write"])
              for r in range(1, 9)))
body = ""
for r in range(1, 9):
    k = 0
    for size in (2048, 4096):
        for links in (1, 2):
            for _ in range(3):
                body += one_test(f"WgapR{r:02d}_w{k:02d}", "unicast_write", size, links, 100000)
                body += one_test(f"WgapR{r:02d}_g{k:02d}", "fused_atomic_inc", 2048, links, 1000)
                k += 1
write("pcs_nx_wgap_x8.yaml", "# cisp row-order C (mk_order_yamls.py): one unicast_write iteration per row, each followed by a\n"
      "# tiny fused_atomic_inc row (1000 packets), 8 repeats.\n", body)


# X6 (coordinator 2026-11-05, yaml only): the num_links discriminator. The NX write rows (as pcs_nx_wonly_x8.yaml, the
# stock L1/L2 alternation) split into L1-only and L2-only blocks, 8 repeats each (48 iterations per arm).
# (An L2 block with only link 0's flows can't be written: expand_link_duplicates copies every sender onto every link
# when num_links > 1, overwriting the yaml's link_id.)
def param_test_links(name, links):
    return param_test(name, ["unicast_write"]).replace("    num_links:\n    - 1\n    - 2\n",
                                                          "    num_links:\n" + "".join(f"    - {l}\n" for l in links))


write("pcs_nx_l1only_x8.yaml", "# cisp X6 (mk_order_yamls.py): NX write rows, num_links 1 only, 8 repeats.\n",
      "".join(param_test_links(f"NeighborExchangeUnicastR{r:02d}", [1]) for r in range(1, 9)))
write("pcs_nx_l2only_x8.yaml", "# cisp X6 (mk_order_yamls.py): NX write rows, num_links 2 only, 8 repeats.\n",
      "".join(param_test_links(f"NeighborExchangeUnicastR{r:02d}", [2]) for r in range(1, 9)))


# X7 idle-gap arm (coordinator 2026-11-05, yaml only): the stock alternation with a near-idle pause between each L2 block
# and the next L1 rows: per repeat, size 2048 (L1 x3, L2 x3), the gap, size 4096 (L1 x3, L2 x3), the gap. The gap is a
# NeighborExchange write at num_links 1 with 1 packet per flow, 10 iterations (~0.3 s of launches with the routers idle).
def size_test(name, size):
    return param_test_links(name, [1, 2]).replace("    size:\n    - 2048\n    - 4096\n", f"    size:\n    - {size}\n")


def gap_test(name):
    return param_test_links(name, [1]).replace("    size:\n    - 2048\n    - 4096\n", "    size:\n    - 2048\n").replace(
        "    num_packets: 100000\n", "    num_packets: 1\n").replace("    iterations: 3\n", "    iterations: 10\n")


write("pcs_nx_gap_x8.yaml", "# cisp X7 (mk_order_yamls.py): NX write rows, L1/L2 alternation with an idle gap after each L2 block.\n",
      "".join(size_test(f"NeighborExchangeUnicastR{r:02d}S2048", 2048) + gap_test(f"GapR{r:02d}a")
              + size_test(f"NeighborExchangeUnicastR{r:02d}S4096", 4096) + gap_test(f"GapR{r:02d}b") for r in range(1, 9)))
