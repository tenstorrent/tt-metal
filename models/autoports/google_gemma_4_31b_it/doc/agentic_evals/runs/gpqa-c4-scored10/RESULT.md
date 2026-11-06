# GPQA C4 scored-ten result

C4/B8 is accepted by the coordinator for the banked frozen10 GPQA workload. Both arms scored9/10 (90%, Wilson95%59.58–98.21%; published reference84.3%, GPUreference83.33%). Scores were reported, not gated. Larger-cohort tails remain open.

Same-current439c runtime, same120-qb2-p04t07 host, same10 IDs/caps/sampling: C2 took938.039050s (0.260566h), C4 took642.445722s (0.178457h). The measured saving is295.593328s = 0.0821092578h, or31.5118%, for this pair. Allten visible response hashes match; both executed2390prefill tokens and56042decode-row steps. C4 required19863decode submissions versus29488C2. Source config differs only in selected concurrency after routing; test expectation differs correspondingly.

The matched executed work supports the measured concurrency benefit rather than a reduced-work explanation. It does not prove full hidden-token equality: rawreasoning IDs/finishreason/usage remain unavailable. One sequential pair is not randomized evidence across hosts or stochastic tails. Allten final answers are complete; hidden cap/loop behavior cannot be categorically excluded from the saved visible texts. Phase counters are host-adapter executedwork, not device timings or rawAPIusage.

Actual effort devicecost 0.761666667h includesC2 .343888889h, C4 .295555556h and wrong-route .122222222h. This exceeds original .508888889h one-arm estimate because the contemporary baseline and failedroute are charged; it remains within5.5hbox. Initial savingguess .065764804125h became 0.0821092578h measured on banked10. No setup speed credit or duplicate charge for separate localcapacitywork.

A20-question proportional saving of 0.1642185155h is only a guess; independentC2twenty continues and is not a C4pair. Full198 remains stopped for nonfit; this result makes no full198fit claim. Keep the accepted change while the larger-tail problem stays open.

Evidence: ../gpqa-current10-c2/audit.json (run42237390829), ../gpqa-c4-scored10-evals/audit.json (run42234229161), and the exact source/config comparison recorded in RESULT.json. No central-file edits or hardware work by this result author.
