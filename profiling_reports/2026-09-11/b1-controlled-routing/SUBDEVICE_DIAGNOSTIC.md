# Manager-free diagnostic candidate

`test_b1_no_manager_candidate.py` removes only explicit subdevice manager creation/load/clear/removal and explicit dispatch subdevice_id. It has not replaced the running worker and has not been executed.

Dispatch core allocation remains identical at the kernel level:

- dispatch/dispatch.cpp:38 uses the active manager's first subdevice when the argument is omitted.
- dispatch/device/dispatch_program_factory.cpp:198 selects all available cores on their first row and sorts by x; lines225–236 select sender groups from that row.
- With2links and default2workers/sender, both full default grid and one-row manager select sender cores(0,0),(3,0) and worker cores(1,0),(2,0),(4,0),(5,0).

What changes is manager scheduling/GO state and profiler subdevice metadata. It is a discriminating diagnostic for manager interaction, not an established bug fix.

The active mFo19z log shows dispatch returned at19:25:03 and tilize kernel compilation followed. Therefore manager clear returned on the host; the observed stall is later in the surrogate tilize path or its queued-device work. Production also clears the manager after dispatch/shared-expert work without an explicit synchronization, so the candidate ordering alone does not establish a defect. Unlike production, this isolated case does not enqueue shared-expert work on the second subdevice.

Source inspection shows manager reset reconfigures worker/GO-mailbox state in SubDeviceManagerTracker::reset_sub_device_state. No specific profiler/subdevice defect has been proven. A completed manager-free run with otherwise identical settings would narrow the hypothesis; it would not alone establish numerical correctness or model performance.
