# Harness file versions

The runs 7a–8a used the harness with **Serbian** messages. The scripts in this folder are the **English** versions, translated on 2026-11-21. Runs 8b–8e ran with these English versions. Apart from the `run_hal_gen.sh` deny-list change described below, the code is the same as in 7a–8a and only messages and comments differ. For the Python files this was checked token by token, with string literals and comments ignored: equal for all 8 files. The other files were never translated.

| File here | md5 of the file here (English) | md5 of the version the runs used |
|---|---|---|
| run_hal_gen.sh | 4cfda0c878d9fe78dd3d04964f98e4ed | same file (8b–8e) |
| run_hal_gen_7a-8a.sh | d53dc454610eb844681b3945d4a6cff7 | b4afc95379d55404e2c4fd839f301ae0 (7a–8a) |
| metrics.py (v2) | 7537af1db2986c154e869f75d156d100 | 9e2e47d7c08ce47f2bd74cd9b66ebdfb (installed 2026-11-18; used by 8a); this file (8b–8e) |
| metrics.py.v1 | 023df838629eef3bb4a5787f902e130a | 05eb0f62d520289d716a73d1ad220278 (7a–7e) |
| make_spec.py | 6d1221281850981367fcadda59cdcd68 | Serbian-message version (local copy 263cf63b2aa6bee92a148843f620fced) |
| codesize.py | c709d9238b89a6f03c7f63a57f943e91 | Serbian-message version (local copy 87336eb2d8c0b72dad368499a0bd4d8d) |
| codestats.py, elfray_tt_tti.py, probe_check.py, test_metrics_synth.py | see MD5SUMS.txt | Serbian-message versions |
| prompt-run6b.md | 411819d28e8e8e5766d81caad6b95b15 | same file (7a–7e) |
| prompt-run6b-cfgcost.md | 3dbea124246f22cae0a3ce1d444d19b7 | same file (8a–8e) |
| hal-settings-B.json | 8f0d3a00c6ad6b6be2aa6da35348b849 | same file (7a–8e) |

**The only difference the agent can see.** `make_spec.py` writes five marker comments into the spec headers the agent reads. In runs 7a–8a these markers were the Serbian wording of the same text; with this version they are English:
- `// [spec] Generated for the HAL experiment (mode B): function bodies and raw encodings have been removed.` (header line)
- `{ /* [spec] body removed */ }`
- `{ /* [spec] removed */ }`
- `/* [spec] removed constant with raw encoding */`
- `// [spec] removed macro with raw instruction`

Runs from 8b on use the English harness, so these markers are English there.

**run_hal_gen.sh (8b–8e) vs run_hal_gen_7a-8a.sh.** The only change is in the step that writes the agent's `hal-settings.json` deny list. On 2026-11-21 the folders on bh-30 were reorganised, and the old top-level names became symlinks. The 8b–8e version therefore counts sibling folders from `/proj_sw/user_dev/$USER` and writes every denied path in all its forms (as given, real path, old symlink name). The goal is unchanged: the agent cannot read other runs, the repo, the results or the tools.

In this copy, one docstring line of `elfray_tt_tti.py` and one error message in both `run_hal_gen*.sh` files were shortened (text only, no code change). Their md5 in `MD5SUMS.txt` therefore differs from the English versions listed above.
