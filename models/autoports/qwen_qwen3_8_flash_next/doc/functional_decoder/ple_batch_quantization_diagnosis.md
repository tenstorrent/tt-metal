# Fresh-prefill packed PLE control, seed 1982

The independent CPU full decoder reproduces request 2, row 0's extra
contribution diagnostic failure. Exact HF versus actual packed-cache lookup
with otherwise unchanged HF arithmetic and BF16 expert weights gives whole
PCC **0.9971231222** and contribution PCC **0.9949601293**. The original device
report `ple_batch4_moe_lifetimes.json` records **0.9971301556** and
**0.9949769974** respectively. This is not a literal contribution pass.
Device BFP8 arithmetic is not needed to reproduce the discrepancy.

The CPU probe preserves probe_batch's seed, constructor order, request lengths
1/2/3/1, prompt/decode random sampling and token-ID sampling. It reads only
selected checkpoint rows and selected words/scales from the complete packed
cache using memory maps. Every selected packed word and scale word is checked
against the production quantizer. Fresh independent literal HF reruns are
bit-exact. The actual complete table remains the source of packed values;
there is no token-specific repair, router override, or changed threshold.

Request 2 row 0 intermediate PCCs are:

| Boundary | Exact HF vs packed-lookup HF PCC |
|---|---:|
| PLE contribution | 0.9982356429 |
| Attention mixed input | 0.9992979169 |
| GDN output | 0.9992981553 |
| MoE mixed input | 0.9994648695 |
| MoE output | 0.9680135846 |

The selected router set changes from expert 434 to expert 20. Thus the largest
observed amplification occurs at routed MoE. Request 2 row 2 also changes one
expert (393 to 220) but its full contribution PCC remains 0.9973678589. All
other tested fresh-prefill rows and all four continuation decode rows in this
CPU comparison meet 0.995. These are CPU controls, not TT acceptance evidence.

Artifacts:

- `ple_batch4_quantization_cpu.json` and `.log`: full/intermediate metrics,
  router sets, input identities, exact rerun and actual-cache parity evidence.
- `/home/ttuser/qwen38-flash/ple_batch4_quantization_cpu.pt`: original inputs,
  tokens, full exact and packed-control outputs/intermediates, and repeated
  literal outputs, retained for device bisection if needed.
- `ple_batch4_control_cpu.json` and `.log`: separate fresh independent control
  through probe_batch's new explicit option, 16 exact cache/quantizer checks,
  literal rerun bit-exact for all four requests and both phases.

CPU control command (completed):

```bash
PYTHONPATH=. python_env/bin/python \
  -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_batch \
  --snapshot /home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa \
  --layer 1 --batch 4 --expert-dtype bfloat8_b \
  --cache-dir /home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide \
  --ple-quantization-control --cpu-only \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ple_batch4_control_cpu.json
```

The optional `--ple-quantization-control` leaves the default strict gate
unchanged. When enabled on hardware it still requires literal HF whole-output
PCC >=0.995 globally and for every row. It also requires device versus the
independent packed-HF control to meet every whole/contribution gate. An extra
fresh-prefill contribution discrepancy is classifiable only when the control
reproduces it at that same aggregate or row. There is no request-index rule.
Literal decode and resumed contribution gates remain strict. Such a run records
`literal_hf_strict_passed: false`, explicit carried rows and a stage-review
requirement; it must not be described as a literal strict pass.

The control includes full cache manifest identity, exact file sizes and
physical-table evidence references. Exact BF16 embedding payload is
102,400,491,520 bytes, exceeding measured device DRAM before other weights;
the selected complete packed representation is 30,720,147,456 bytes. These
physical constraints explain why a packed control is relevant, but do not
alone authorize accepting a failed gate. Final independent stage review must
assess this explicit fresh-prefill diagnostic classification. No hardware run
or device-to-control PCC is claimed by this report.
