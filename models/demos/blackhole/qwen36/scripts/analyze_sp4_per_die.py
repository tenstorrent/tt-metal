#!/usr/bin/env python3
"""Per-die, per-layer split of a 4-die SP prefill Tracy ops CSV: compute vs comm vs SDPA time.
Segments each device's last pass into GDN/ATTN layers plus a tail, from op-code and output-shape patterns.
"""
import re
import sys

import numpy as np
import pandas as pd

KERNEL_COL = "DEVICE KERNEL DURATION [ns]"
DIE_OF_DEVICE = {3: 0, 2: 1, 1: 2, 0: 3}
DEVICE_OF_DIE = {v: k for k, v in DIE_OF_DEVICE.items()}
LN_SHAPE = (1, 1, 1024, 2048)

OUT_SHAPE_COLS = [
    "OUTPUT_0_W_PAD[LOGICAL]",
    "OUTPUT_0_Z_PAD[LOGICAL]",
    "OUTPUT_0_Y_PAD[LOGICAL]",
    "OUTPUT_0_X_PAD[LOGICAL]",
]


def lastdim(cell):
    m = re.match(r"(\d+)\[", str(cell))
    return int(m.group(1)) if m else None


def out_shape(row):
    return tuple(lastdim(row[c]) for c in OUT_SHAPE_COLS)


def out_xdim(row):
    return lastdim(row["OUTPUT_0_X_PAD[LOGICAL]"])


def is_comm(op_code):
    return bool(re.search(r"(Send|Recv)", str(op_code)))


def kernel_ms(rows):
    return rows[KERNEL_COL].sum() / 1e6


def find_embedding_rows(df):
    mask = df["OP CODE"].str.contains("Embeddings", na=False)
    return df.index[mask].tolist()


def split_passes(device_df):
    emb_idx = find_embedding_rows(device_df)
    if not emb_idx:
        return [device_df], emb_idx
    passes = []
    for k, e in enumerate(emb_idx):
        end = emb_idx[k + 1] - 1 if k + 1 < len(emb_idx) else device_df.index[-1]
        passes.append(device_df.loc[e:end])
    return passes, emb_idx


def find_layer_starts(pas):
    ln_idx = [i for i in pas.index[pas["OP CODE"] == "LayerNormDeviceOperation"] if out_shape(pas.loc[i]) == LN_SHAPE]
    last_idx = pas.index[-1]
    starts = []
    for i in ln_idx:
        for j in range(i + 1, min(i + 4, last_idx + 1)):
            if pas.loc[j, "OP CODE"] == "MatmulDeviceOperation":
                xd = out_xdim(pas.loc[j])
                if xd == 8224:
                    starts.append((i, "GDN"))
                    break
                elif xd == 5120:
                    starts.append((i, "ATTN"))
                    break
    return starts, ln_idx


def find_mlp_end(pas, layer_start_idx, ln_idx):
    last_idx = pas.index[-1]
    mlp_ln = min(x for x in ln_idx if x > layer_start_idx)
    mm6144 = [
        i
        for i in range(mlp_ln + 1, last_idx + 1)
        if pas.loc[i, "OP CODE"] == "MatmulDeviceOperation" and out_xdim(pas.loc[i]) == 6144
    ]
    search_from = mm6144[1] + 1 if len(mm6144) >= 2 else mlp_ln + 1
    down_idx = next(
        i
        for i in range(search_from, last_idx + 1)
        if pas.loc[i, "OP CODE"] == "MatmulDeviceOperation" and out_xdim(pas.loc[i]) == 2048
    )
    if down_idx + 1 <= last_idx and pas.loc[down_idx + 1, "OP CODE"] == "BinaryNgDeviceOperation":
        return down_idx + 1
    return down_idx


def build_layers(pas):
    starts, ln_idx = find_layer_starts(pas)
    layers = []
    for k, (s, typ) in enumerate(starts):
        end = starts[k + 1][0] - 1 if k + 1 < len(starts) else find_mlp_end(pas, s, ln_idx)
        layers.append(dict(layer=k, type=typ, start=s, end=end))
    last_idx = pas.index[-1]
    tail_start = layers[-1]["end"] + 1
    tail = pas.loc[tail_start:last_idx] if tail_start <= last_idx else pas.iloc[0:0]
    return layers, tail


def layer_stats(pas, layer):
    rows = pas.loc[layer["start"] : layer["end"]]
    total = kernel_ms(rows)
    comm = kernel_ms(rows[rows["OP CODE"].apply(is_comm)])
    sdpa = kernel_ms(rows[rows["OP CODE"] == "SDPAOperation"])
    return dict(
        layer=layer["layer"],
        type=layer["type"],
        total_ms=total,
        comm_ms=comm,
        compute_ms=total - comm,
        sdpa_ms=sdpa,
        ops=len(rows),
    )


def print_layer_table(die, dev, per_layer, tail):
    print(f"\n{'-'*100}\nDie {die} (DEVICE ID {dev}) per-layer table (LAST pass)\n{'-'*100}")
    print(f"{'layer':>5} {'type':>5} {'total ms':>10} {'comm ms':>9} {'compute ms':>11} {'sdpa ms':>9} {'ops':>5}")
    for r in per_layer:
        print(
            f"{r['layer']:>5} {r['type']:>5} {r['total_ms']:>10.4f} {r['comm_ms']:>9.4f} "
            f"{r['compute_ms']:>11.4f} {r['sdpa_ms']:>9.4f} {r['ops']:>5}"
        )
    print(f"{'tail':>5} {'--':>5} {kernel_ms(tail):>10.4f} {'':>9} {'':>11} {'':>9} {len(tail):>5}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_sp4_per_die.py <ops_perf_results.csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path, low_memory=False)
    device_df_all = df.loc[df[KERNEL_COL].notna()].copy()
    dev_ids = sorted(int(d) for d in device_df_all["DEVICE ID"].dropna().unique())
    print(f"CSV: {csv_path}")
    print(f"Devices found: {dev_ids}")
    print(f"Die mapping used: {DIE_OF_DEVICE}")

    per_die = {}
    for dev in dev_ids:
        d = device_df_all[device_df_all["DEVICE ID"] == dev].reset_index(drop=True)
        passes, emb_idx = split_passes(d)
        die = DIE_OF_DEVICE.get(dev, dev)
        last = passes[-1].reset_index(drop=True)
        layers, tail = build_layers(last)
        per_layer = [layer_stats(last, layer) for layer in layers]
        per_die[die] = dict(
            dev=dev,
            n_passes=len(passes),
            last_pass=last,
            layers=layers,
            per_layer=per_layer,
            tail=tail,
        )
        print(
            f"\nDEVICE ID {dev} (die {die}): {len(passes)} pass(es) found (embedding rows: {len(emb_idx)}); "
            f"last pass rows={len(last)}; layers found={len(layers)}"
        )

    # --- Task 2: per-layer tables for die 0 and die 3 ---
    for die in (0, 3):
        info = per_die[die]
        print_layer_table(die, info["dev"], info["per_layer"], info["tail"])

    # --- Task 1: die-3 tail op list ---
    die3 = per_die[3]
    print(f"\n{'-'*100}\nDie 3 (DEVICE ID {die3['dev']}) tail op list (after layer-23 MLP-pattern end)\n{'-'*100}")
    print(f"tail rows: {len(die3['tail'])}, tail kernel ms: {kernel_ms(die3['tail']):.4f}")
    print(f"{'OP CODE':<40}{'kernel us':>12}")
    for _, r in die3["tail"].iterrows():
        print(f"{str(r['OP CODE'])[:40]:<40}{r[KERNEL_COL]/1e3:>12.2f}")

    # --- Task 2: summary lines for all four dies ---
    print(f"\n{'-'*100}\nSummary (all four dies, LAST pass)\n{'-'*100}")
    for die in sorted(per_die):
        info = per_die[die]
        pl = info["per_layer"]
        gdn = [r for r in pl if r["type"] == "GDN"]
        attn = [r for r in pl if r["type"] == "ATTN"]
        gdn_excl0 = [r for r in gdn if r["layer"] != 0]
        mean_compute_gdn = np.mean([r["compute_ms"] for r in gdn_excl0])
        mean_compute_attn = np.mean([r["compute_ms"] for r in attn])
        mean_comm_gdn = np.mean([r["comm_ms"] for r in gdn])
        mean_comm_attn = np.mean([r["comm_ms"] for r in attn])
        tail_ms = kernel_ms(info["tail"])
        tail_ops = len(info["tail"])
        pass_kernel_sum = kernel_ms(info["last_pass"])
        print(
            f"die {die} (dev {info['dev']}): mean_compute_GDN(excl L0)={mean_compute_gdn:.4f} ms | "
            f"mean_compute_ATTN={mean_compute_attn:.4f} ms | mean_comm_GDN={mean_comm_gdn:.4f} ms | "
            f"mean_comm_ATTN={mean_comm_attn:.4f} ms | tail={tail_ms:.4f} ms/{tail_ops} ops | "
            f"pass_kernel_sum={pass_kernel_sum:.4f} ms"
        )

    # --- SDPA op durations per die ---
    print(f"\n{'-'*100}\nSDPA op durations per die (LAST pass, 6 expected each)\n{'-'*100}")
    for die in sorted(per_die):
        info = per_die[die]
        sdpa_rows = info["last_pass"]
        sdpa_rows = sdpa_rows[sdpa_rows["OP CODE"] == "SDPAOperation"]
        durs = (sdpa_rows[KERNEL_COL] / 1e3).tolist()
        print(f"die {die} (dev {info['dev']}): n={len(durs)} durations_us={[f'{x:.2f}' for x in durs]}")

    # --- Task 3: die-3 specifics ---
    print(f"\n{'-'*100}\nTask 3: die 3 (DEVICE ID {die3['dev']}) specifics\n{'-'*100}")
    layer0 = die3["per_layer"][0]
    print(
        f"Layer-0 GDN total (kernel ms): {layer0['total_ms']:.4f} ms "
        f"(compute={layer0['compute_ms']:.4f} ms, comm={layer0['comm_ms']:.4f} ms, ops={layer0['ops']})"
    )

    last = die3["last_pass"]
    emb_row_idx = last.index[last["OP CODE"].str.contains("Embeddings", na=False)][0]
    recv_rows = last[last["OP CODE"].apply(is_comm)]
    first_recv_idx = recv_rows.index[0]
    first_recv_dur_ns = last.loc[first_recv_idx, KERNEL_COL]
    print(
        f"First recv op in pass: row idx {first_recv_idx}, OP CODE={last.loc[first_recv_idx,'OP CODE']}, "
        f"duration={first_recv_dur_ns/1e3:.3f} us ({first_recv_dur_ns/1e6:.4f} ms) -- pipeline-fill wait"
    )

    first_layer_start = die3["layers"][0]["start"]
    between = last.loc[emb_row_idx + 1 : first_layer_start - 1]
    print(
        f"Ops between Embeddings row (idx {emb_row_idx}) and first layer's LayerNorm (idx {first_layer_start}): "
        f"{len(between)} ops, kernel_ms={kernel_ms(between):.4f}"
    )
    for _, r in between.iterrows():
        print(f"   {r['OP CODE']}: {r[KERNEL_COL]/1e3:.3f} us")


if __name__ == "__main__":
    main()
