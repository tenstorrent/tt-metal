Goal: Enable all paths from Keraunos CCEs. Specific paths identified so far are:
- Processor read & write to own SRAM
- Processor read & write to SRAM in different HSIO tiles
- Processor read & write to each GDDR location
- Enable support for CCE1. CCE1 uses a different SRAM address compared to CCE0. It's SRAM base is 4MiB, and to access CCE0 SRAM uses base 0MiB. For CCE0, it accesses CCE1 SRAM using base 4MiB. To make this easier for users, I want a set of device-side APIs that automatically do this for users. Look at how fabric is implemented in Blackhole to come up with what APIs will be useful and create tests that demonstrate how a Keraunos fabric implementation (using the 'Quasar' architecture in tt-metal) would use these APIs.

Look at tt-metal fabric support for Blackhole and identify if any other paths are likely required.
Most of this currently appears to be working and should have existing tests. Attempts to have Keraunos read from GDDR either hangs or reads zero back. What has been tried so far has been captured in `keraunos_mimir_gddr_read.md`.

You may make changes to either the grendelemulation model, chippy scripts or tt-metal. However, as an eventual goal we want to use existing grendelemulation APIs, chippy APIs and similar tt-metal APIs to what fabric on Blackhole currently uses. If it is not possible to get these paths working through those APIs, you may still proceed, but you should attempt to map working flows back to these APIs when possible before completing the session. Your development process can include creating direct chippy tests to verify the firmware and path operation before attempting to make these tests work in tt-metal - I leave that decision up to you.

Issues have been found with both the grendelemulation models and chippy scripts/firmware in the past, including specifically related to these paths. If you cannot find a solution with reasonable effort, create a small chippy testcase/script that replicates the failure, and stop attempting work on that path until I can get feedback from the SysEng experts. This will either give us a concrete, simple example to share with the SysEng experts for further debug, or tell us that the path works in chippy and the error exists in the tt-metal stack.

Before starting the implementation & debug work, please do the following:
- Come up with a concrete plan of how you will attempt to solve this problem and save it to a .md file for my review
- Let me know if there are any decisions needed, if anything is ambiguous that requires my input, or if you have any questions for me

When doing the implementation & debug, save what choices, discoveries and changes you made at each major step so we can review later if necessary. The local repositories are all clean and up to date so you can commit changes at useful/major stages to make it easier to follow the code changes later. The edited files in tt-metal are a failed attempt to get the processor reading GDDR path working so can be discarded if it is not useful.
