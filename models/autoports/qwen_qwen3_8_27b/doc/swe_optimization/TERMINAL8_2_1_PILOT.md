# Terminal Bench 2.1 eight-task pilot

QB2 run 40695765783 was a Terminal Bench **2.1** pilot with Terminus-2,
eight concurrent tasks, B32 serving, 2,048-token cache blocks, grouped,
masked and flat prefill enabled, medium reasoning effort, seed 1234,
temperature 1, top-p 0.95 and top-k 20. It used image
`sha256:0611f1029baae3c347440a84f511ec8d156b1d3cb953e0471324910e8b17388d`
and tt-inference-server `1b80221ffb59c2b47da3e6f40b5665aff0b7dc48`.
The exact eight names and dataset digest are in the archived Harbor config.

| Outcome | Count |
|---|---:|
| Solved | 2/8 (25%) |
| Normal completion, unsolved | 3/8 |
| Agent timeout, unsolved | 3/8 |

The solved tasks were `fix-code-vulnerability` and
`nginx-request-logging`. `dna-assembly`, `feal-linear-cryptanalysis`, and
`path-tracing` reached the 10,800-second agent limit. The Harbor phase lasted
10,840.135 seconds (3.011 hours), dominated by those timeouts. The model's
server-interval metrics report 3,346,432 cached of 3,640,556 prompt tokens
(91.92%), 332,240 generated tokens and zero preemptions. The cache ratio is a
server prompt-token metric, not the eligible-prefix denominator used by the
separate 212-request synthetic study. The metrics file warns that engine
identity was not verified by the API process guard, so these counters are
supporting evidence, not a request-joined ledger.

The workflow succeeded, all eight result records were recovered, the exact
two-file HTTP/worker audit export passed stable reads, and owned cleanup
reported no unresolved resources. Evidence remains outside Git in
`/home/mvasiljev/build/qwen38-swe/terminal8-40695765783/`.

There is no matched Terminal8 control or GPU reference in this pilot. Its
2.1 roster must not silently stand in for the Gemma Terminal89 2.0 roster;
select the benchmark version before running Terminal89. This pilot does not
support an accuracy or time projection for the full 89 tasks.
