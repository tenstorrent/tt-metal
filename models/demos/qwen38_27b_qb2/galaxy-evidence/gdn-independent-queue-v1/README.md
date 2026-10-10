# Independent recurrence queue

At 21:50:11 UTC on October 10 the following persistent user services were active:

- `qwen38-compact-long-horizon-v6-20261010.service`, invocation `3e818cd8174c45b2a50af75aad61f7c2`.
- `qwen38-gdn-resident-hardware-v3-20261010.service`, invocation `7effa827159e4f9cb36e5023d6630b0b`.
- `qwen38-gdn-compact-gates-v3-20261010.service`, invocation `9d7889e59af942ef8e45967635e28d95`.

The first follows the successful projection sweep, then the others run in order.
Their frozen sources, accuracy gates, hardware lock and resource/time limits are
unchanged. They survive disconnect, not reboot. A live unit is not a passed test.
The separate failed prefill experiment remains unpromoted for investigation.

Before requeueing, exact old invocation identities, empty control groups,
unstarted dependency-failure receipts and all source hashes were checked.
An initial staging attempt omitted a Python import and stopped before launching
any unit. Its prepared v5 control directory is retained; corrected staging uses
a fresh v6 directory. `stage-recovery.py.gz` and launch/audit receipts record the
recovery. No native install, firmware or checkpoint was changed.
