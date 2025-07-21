# set SFPI release version information

sfpi_version=v6.13.0-gcc
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=0ccdf2b691e66fbb534cfffd699d00aa
sfpi_x86_64_Linux_deb_md5=cb0682d2da7bc2cac8ca3eff759897cc
sfpi_x86_64_Linux_rpm_md5=aaac3235d76614eef65d2b14f0cbd94d
