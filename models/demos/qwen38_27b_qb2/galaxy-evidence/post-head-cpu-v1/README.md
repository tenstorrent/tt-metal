# CPU reference and image-runtime follow-up

`qwen38-post-head-cpu-v1-20261009.service` was launched on `10.228.203.98`
at 05:50:50 UTC, Oct 9, and confirmed live with PID 1209320. It is waiting,
not yet running its image or model-reference stages. It waits for both the
native experiment queue and the BFP8/HiFi2-head qualification to finish and
confirm worker shutdown. Exact service invocations and source hashes are
recorded in `status-at-launch.json`.

The work is intentionally serialized after hardware measurements so CPU
inference and Docker import do not contaminate their client timings. A source
change, replaced service, failed dependency or incomplete cleanup stops the
follow-up rather than inferring that the machine is idle.

1. Copy the verified OCI image from `.34` to `.98`, verify its full archive
   checksum, load it into Docker, then run source/version/import verification
   and the standard TTIS entrypoint's `--help` in bounded containers with no
   network or accelerator devices. These checks cannot qualify inference.
2. Run `demo/probe_hf_head.py` against the existing pinned checkpoint on CPU.
   The reference loads all weights in BF16 with Transformers 5.12.1, rejects
   missing/unexpected/mismatched keys, and retains eight reference steps for
   the same short public prompt as G0. Compare FP32 head projections using
   original, host-round-tripped BFP4 and BFP8 weights on the same HF hidden
   states. Report logit RMS error, KL divergence, top-1 and top-20 agreement.

The numerical probe excludes device LoFi/HiFi2 arithmetic and uses full BF16
reference decoder weights. It can identify head-weight sensitivity, but it
does not predict a GPQA score or establish the cause of current failures.
The actual full-model BFP8/HiFi2-head GPQA remains the deciding experiment.

The outer service has a 20-hour limit, an 18-hour dependency wait, 160-GiB RAM
and eight CPU quota. The image stage has a 2,600-second bound; the CPU reference
has 1,800 seconds. It survives client disconnect, not reboot. Source snapshots
are retained here exactly; source hashes refer to the executable host copies.

The image transfer uses a temporary TLS server on `.34:18443`, serving only
the single checksum-addressed archive to client `.98`. Its public certificate
was copied through authenticated SSH and is explicitly trusted by the client;
no private key or registry credential was transferred. The server is bounded
to 20 hours and is owned by `qwen38-serve-image-v2-20261009.service`.

The source host has about 26 GiB free, while the conservative load budget is
31.2 GiB including an 8-GiB reserve. It was not loaded there. The receiver
checks available disk before downloading and again before import; it does not
delete existing images. Results will be in `release-image-v1` and
`hf-head-reference-v1` beneath the task artifact directory. Neither stage has
passed yet, and the existing artifact remains unqualified.

Additional hardware timing jobs should wait for this CPU follow-up to finish.
