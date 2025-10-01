# set SFPI release version information

sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi_\([-.0-9a-zA-Z]*\)_\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_version=\2'"\n"'sfpi_\3_\4_md5=\1/' *-build/sfpi_*.md5 | sort -u
sfpi_version=7.3.0-replay-29538
sfpi_x86_64_deb_md5=6b9a9f26447de853e2501f99389b554e
sfpi_x86_64_rpm_md5=157c2c71546c7dcb8679f059f6f4519d
sfpi_x86_64_txz_md5=b994281c766d808970cdda7f209fbc22
