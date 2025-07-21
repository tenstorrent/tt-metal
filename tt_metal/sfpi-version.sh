# set SFPI release version information

sfpi_version=v6.13.0-gcc
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=2ed75d9b9871c77150bcf8724e490ec7
sfpi_x86_64_Linux_deb_md5=242fdf000c300af00441713703e2e343
sfpi_x86_64_Linux_rpm_md5=3d7c557b4d40ced7a7b79f3f7a8441b5
