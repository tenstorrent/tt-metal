# set SFPI release version information

sfpi_version=v6.12.0-insn-synth
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=8234657e9507bd1860117856bb3fc178
sfpi_x86_64_Linux_deb_md5=7fc6128615c0106f08972d447e424b9d
sfpi_x86_64_Linux_rpm_md5=3416663ddfc4824e0fd54740f4870d74
