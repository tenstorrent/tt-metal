# Host sampling and explicit device sampler reloads

Source audit of pinned `vllm-tt-plugin` commit
`042a94863ad27d88207bcf71abdf67122c6f4c27` found no serving-contract gap here.
`src/vllm_tt_plugin/async_decode.py:464–483` makes host sampling authoritative
(`reload_inputs=True`) and computes both sampler flags from
`device_sampling and transition`. Therefore host sampling always sends
`reload_sampling_params=False` and `reset_sampling_state=False`.
Lines 1373–1394 supply generator sampling parameters only for device sampling;
host sampling retains those parameters in the plugin's host sampler.

Direct low-level callers requesting host logits must follow the same contract:
`sample_on_device=False`, authoritative host token/position inputs, and both
device sampler reload/reset flags false. `reload_sampling_params=True` explicitly
requests a device sampler configuration update and therefore retains device
limits, including top-k at most 32; selecting host logits does not broaden that
separate command. Host-owned top-k50, penalties, and logprob settings belong to
the caller's host sampler and do not need to be uploaded to the device sampler.

No generator or probe change was made. The apparent top-k50 rejection combined
an explicit device-configuration command with host-only sampling parameters;
the pinned plugin does not construct that combination. Existing high-level
`probe_full_generator --host-compatibility` exercises its separately supported
host callback path with top-k50. No hardware ran for this source audit, and the
currently running B1 trace remains on the unchanged generator revision.
