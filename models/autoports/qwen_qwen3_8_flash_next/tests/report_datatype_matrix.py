"""Index material same-dtype fidelity comparisons from full-model evidence."""
import json
from pathlib import Path

OUT = Path('models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep')
PAIRS = [
    ('GDN inner projections', 'BFP4', 'gdn_inner_bf4_lofi', 'gdn_inner_bf4_hifi2'),
    ('GDN projections', 'BFP8', 'baseline_policy', 'gdn_bf8_hifi2'),
    ('QSA decode projection', 'BFP4', 'baseline_policy', 'qsa_bf4_hifi2'),
    ('QSA decode projection', 'BFP8', 'qsa_bf8_lofi', 'qsa_bf8_hifi2'),
    ('QSA inner output projections', 'BFP4', 'qsa_output_inner_bf4_lofi', 'qsa_output_inner_bf4_hifi2'),
    ('Shared inner experts', 'BFP4', 'shared_inner_bf4_lofi', 'shared_inner_bf4_hifi2'),
    ('Shared experts in BFP8 projection family', 'BFP8', 'all_projection_bf8_lofi', 'all_projection_bf8_shared_hifi2'),
    ('HC inner projections', 'BFP4', 'hc_inner_bf4_lofi', 'hc_inner_bf4_hifi2'),
    ('LM head', 'BFP4', 'head_bf4_lofi', 'head_bf4_hifi2'),
    ('LM head', 'BFP8', 'head_bf8_lofi', 'head_bf8_hifi2'),
    ('Routed expert gate/up/down', 'BFP4', 'baseline_policy', 'experts_bf4_hifi2'),
]


def main():
    rows = {r['config_id']: r for r in json.loads((OUT / 'sweep_results.json').read_text())}
    winner = max((r for r in rows.values() if r['status'] == 'pass'),
                 key=lambda r: r['traced_teacher_forcing_tps'])
    comparisons = []
    lines = ['# Material fidelity comparisons', '',
             'Each speed is full-model traced teacher forcing, median of three B1 prompt203/gen100 runs. Accuracy is teacher-forcing top-1; full prefill/top-5/top-100 gates remain in sweep_results.json. Shared BFP8 is compared within the same BFP8 projection family. QSA singleton packed projections inherit the explicit qsa_decode fidelity rule.', '',
             '| Group | Weight dtype | LoFi config | LoFi top-1 / t/s/u | HiFi2 config | HiFi2 top-1 / t/s/u | Decision / reason |',
             '|---|---|---|---:|---|---:|---|']
    for group, dtype, low, high in PAIRS:
        pair = [rows[low], rows[high]]
        assert all(r['traced_teacher_forcing_tps'] is not None for r in pair)
        faster = max(pair, key=lambda r: r['traced_teacher_forcing_tps'])
        decision = (f"Both gates pass; {faster['config_id']} has the higher pair median. "
                    f"Final policy: {winner['config_id']}; interaction trials decide the combined policy.")
        comparisons.append(dict(group=group, weight_dtype=dtype, lofi=pair[0], hifi2=pair[1], decision=decision))
        lines.append(f"| {group} | {dtype} | `{low}` | {pair[0]['top1']:.0%} / {pair[0]['traced_teacher_forcing_tps']:.6f} | `{high}` | {pair[1]['top1']:.0%} / {pair[1]['traced_teacher_forcing_tps']:.6f} | {decision} |")
    lines += ['', 'HC/QSA-output BFP8 fidelity changes additionally appear in the matched-weight qsa_bf8_lofi/canonical_hifi2 policy family; those family measurements change multiple fidelities and do not isolate individual group speedups. PLE BFP4 tables are lookups, not matmuls. Router/routing numerics retain the sensitive baseline contract. All considered BFP4 matmul groups above include real full-model LoFi evidence.', '']
    (OUT / 'group_comparisons.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    (OUT / 'group_comparisons.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
