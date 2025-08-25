# set SFPI release version information

sfpi_version=v6.18.0-constraints
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=cf4387f468a238d9ba9b98744a5f5c22
sfpi_x86_64_Linux_deb_md5=5a3885a52b23830744a9582c52c5cddc
sfpi_x86_64_Linux_rpm_md5=5e7b4718d983bcb5c57cc49455add7ee
