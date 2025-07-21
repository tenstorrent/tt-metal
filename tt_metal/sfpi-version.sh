# set SFPI release version information

sfpi_version=v6.17.0-gcc
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=d336f316860463268facee27a6a85ead
sfpi_x86_64_Linux_deb_md5=b86044c48be2c5fb9490a95b655fffff
sfpi_x86_64_Linux_rpm_md5=255c67b4b8e15e1ae8498bc7bdf24a6c
