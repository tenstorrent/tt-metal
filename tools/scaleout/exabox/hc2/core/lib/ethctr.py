# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# copied from an earlier tool for run.sh's evidence logging (lib/heavy_logs.sh)
# read-only ETH link counters from the BH ERISC base firmware's boot_results (L1 0x7CC00, tt-umd 0.9.12
# blackhole_eth.hpp): port/train status, eth_live_status (retrain_count, rx_link_up, frames/bytes tx/rx incl. bad FCS and
# dropped, FEC corrected/uncorrected codewords), serdes cdr_unlocked_cnt, and our ctl/status block when --ctl is given.
#   ethctr.py <devnode>:<translated x>,<y> [...] [--ctl ADDR]    -> one JSON line per core
import json, sys
import tt_umd as u

BOOT = 0x7CC00
LIVE = BOOT + 128 * 4
SERDES = BOOT + 32 * 4
NAMES64 = ["frames_txd", "frames_txd_ok", "frames_txd_badfcs", "bytes_txd", "bytes_txd_ok", "bytes_txd_badfcs",
           "frames_rxd", "frames_rxd_ok", "frames_rxd_badfcs", "frames_rxd_dropped", "bytes_rxd", "bytes_rxd_ok",
           "bytes_rxd_badfcs", "bytes_rxd_dropped", "corr_cw", "uncorr_cw"]


def words(d, x, y, a, n):
    b = d.noc_read(x, y, a, 4 * n)
    return [int.from_bytes(b[4 * i:4 * i + 4], "little") for i in range(n)]


args = [a for a in sys.argv[1:] if not a.startswith("--")]
ctl = None
if "--ctl" in sys.argv:
    ctl = int(sys.argv[sys.argv.index("--ctl") + 1], 0)
    args = [a for a in args if a != sys.argv[sys.argv.index("--ctl") + 1]]
devs = {}
for spec in args:
    n, xy = spec.split(":")
    x, y = (int(v) for v in xy.split(","))
    if n not in devs:
        d = u.TTDevice.create(int(n)); d.init_tt_device(); devs[n] = d
    d = devs[n]
    st = words(d, x, y, BOOT, 4)
    live = words(d, x, y, LIVE, 40)
    ser = words(d, x, y, SERDES, 64)
    r = {"core": spec, "postcode": hex(st[0]), "port_status": st[1], "train_status": st[2],
         "retrain_count": live[0], "rx_link_up": live[1], "cdr_unlocked_cnt": ser[42], "cdr_unlock_transitions": ser[43],
         "anlt_retry_cnt": ser[8]}
    for i, k in enumerate(NAMES64):
        r[k] = live[8 + 2 * i] | (live[9 + 2 * i] << 32)
    if ctl is not None:
        c = words(d, x, y, ctl, 12) + words(d, x, y, ctl + 0x40 + 64 * 16, 8)
        r["ctl"] = {"hello": c[0], "rxseq": hex(c[4]), "ack": hex(c[8]),
                    "status": dict(zip(["epoch", "phase", "tx", "rx", "acked", "avail", "bad", "last_hdr"], c[12:20]))}
    print(json.dumps(r), flush=True)
