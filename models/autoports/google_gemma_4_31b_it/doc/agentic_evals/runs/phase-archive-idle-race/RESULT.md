# Idle phase archive repair

Accepted minimal model-owned repair, source-only commit `ab2730c27e557b09b2eafe3494e47790adc210aa` on11a387. Idle automatic export now skips only when no events exist and semantic metadata matches the last successful write. Manual drains, event chunks and changes to pending/error/loss/reset/finalbuffer state still export. No collector, plugin or harness changes. Actual CI artifact confirmation remains pending.

Original host reproduction grew4265bytes while idle/.25s and tar returned1/file-changed; candidate identical fixture stayed stable and tar returned0. Live native pair against provenance-checked accepted baseline matched all3semantic/tool/usage/token responses,583expected decodeIDs,200overlapsteps and3nonidentity remaps. Three prefillfirsttokens excluded and2extraactive samples not emitted. Zero loss/errors/pending.

Live file stayed byte-size/mtime stable for3.2s across1s automaticcycles, after periodic completion with no finalmanualflush. Tar returned0, archivedbytes exact, all18JSONL records parsed, finalbuffered0/pending0/errors[]/lossfalse. Fresh0/0idle then graceful closure confirmed owners empty. Cost0.059046271deviceh versus0.12forecast, within0.25box; baseline reused at0newcost. No speedup claimed.

Validation:6newhost tests plus13recorder/15retention tests and all applicable hooks passed. Source-only publication bytes exactly match live patch. AutoDebug isolated inspection corroborated source/log chain, followed by AutoFix reproduction and focused checks.

The stockcollector rejects the archive when tar reports a changedfile; sourcefiles are not thereby deleted. The specific oldCI colliding append could have been finalgenuinework rather than emptyidle status. This repair removes recurringidle churn but does not make collection atomic against newwork, finalstatus, resets or manualdrains. Existing cap/diskfailure/partialwrite limitations remain. It does not recover oldGPQA20 phases or bank that run. Root integrates central records and performs publication/CI confirmation.
