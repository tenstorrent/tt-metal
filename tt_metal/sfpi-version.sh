# set SFPI release version information

sfpi_version=v6.12.0-instrn_buffer
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=0c215d05882990c9f9ccc6b391c496b0
sfpi_x86_64_Linux_deb_md5=4bd958025b4fbe0060722dfef7c3acdc
sfpi_x86_64_Linux_rpm_md5=da92f034f680396faf9793803c43a09c
