# Native loop-onset capture result

Complete native prefixes captured for both requests, including firstprefill tokens. Request0/actualdoc7 has481prompt+4616generated tokens; request1/actualdoc20 has258prompt+4617generated tokens. Pinned tokenizer full-ID equality with each same-request SSE prompt independently confirms identity. This corrects the earlier swapped phase-length assignment, without any evidence of runtime request crossing.

Every native generated token afterindex0 matches both the prior retained replay chain and current retained chain:4615/4616 exact comparisons, no mismatches or missing IDs. Current retention has twoadditional physical samples perrequest after HTTPdisconnect; these are not claimed emitted.4620submit/completionstepIDs match; no associationambiguity, finalpending/errors/loss.

Both captured suffixes are period2; native generated onsetindices4104(doc7) and3179(doc20). Coordinator captured sufficient tail evidence then closed only ownedHTTPconnections, preserving126976output/14400srequest settings. Fresh0/0idle preceded server closure. There is no final finishreason/fullcap completion/accuracy score. Complete captured native prefixes are now available for a causal distribution discriminator; exact overlap does not establish originalCI hidden trajectory identity.

Actualdoc7 case source is private native-0.json. Prefix prompt+generated[:4104] has4585tokens and predicts generated[4104]; after16repeat tokens prefix[:4120] has4601tokens. No splicing or inventedfirsttoken. Sourcepackaging received these paths, offset convention and coverage confirmation.

Client elapsed312.884784s including readiness; queue cost0.090026862deviceh versus0.15hforecast, within0.25hbox. Accepted runtime source hashes match11a387692d2b434198e79dd92fbb8ed308c3ca47. No runtime change, profiler or score claim. Raw prefixes/SSE stayprivate; compact evidence references are in RESULT.json.
