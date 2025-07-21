# set SFPI release version information

sfpi_version=v6.13.0-gcc
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=15058bc02ff2df726b710150b9815315
sfpi_x86_64_Linux_deb_md5=ca496294c982cd1661144f43ffb62432
sfpi_x86_64_Linux_rpm_md5=154ce6de45aae2c081b1ef1d7edd5ac6
