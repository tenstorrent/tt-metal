# set SFPI release version information

sfpi_version=v6.12.0-insn-synth
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=2f7ca3611da5cb30fbec0ec111f1f6d3
sfpi_x86_64_Linux_deb_md5=f28f9bd0a37925480e3845404fd52e86
sfpi_x86_64_Linux_rpm_md5=ec3f9d80c2f0fdbec520f185ef88c806
