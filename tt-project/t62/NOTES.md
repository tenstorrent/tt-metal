# t62
blx03 job 030 submitted 2026-10-01 10:32 (run58.sh, unpatch A/B). Log on blx03: /var/log/tt-device-broker/2026-10-01_103213_030.log
Next: check status (ssh g14blx03 tt-device-mcp status -j 030); grep T58_AB in log; pass = bit_identical_chwt=True and bit_identical_yuv=True.
Copy run58.log back to tt-project/t62/, clean /var/tmp/fasth3/t58 and ~/fasth3/t58 on blx03. If pass: cherry-pick b9312ff3061 onto ttp/t48-ltx25-integrated, ttp push.
If chip drop/reboot during job 030: stop all device work, report.
