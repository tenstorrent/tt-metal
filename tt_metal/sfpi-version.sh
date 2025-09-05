# set SFPI release version information

sfpi_version=v6.19.0-constraints
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=d6541b9d1b480ca071cc7011fc4d5a1b
sfpi_x86_64_Linux_deb_md5=8ad31c00ced6c2932c18f4a0477779b8
sfpi_x86_64_Linux_rpm_md5=f40c23a8c1ac215ff186e21a554eb4a3
