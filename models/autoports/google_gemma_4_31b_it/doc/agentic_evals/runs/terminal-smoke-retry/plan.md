# Terminal smoke retry registration

Goal: switch the dedicated TTI publication configuration to the identical frozen Terminal smoke2 after the infrastructure failure, disabling SWE and retaining GPQA disabled. Preserve all native parser fixes and revision pins from da219dd56a241bb5314ae1da4b87104b84b0da15.

Make only model-specific CI_NIGHTLY limit changes; run focused registration checks and publication hooks. No commit, publication, dispatch or device work. Device-hour budget zero. Fabric diagnosis belongs to the other effort; this registration does not assert hardware recovery or successful evaluation.
