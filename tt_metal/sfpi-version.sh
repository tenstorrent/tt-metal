# set SFPI release version information

sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi_\([-.0-9a-zA-Z]*\)_\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_version=\2'"\n"'sfpi_\3_\4_md5=\1/' *-build/sfpi_*.md5 | sort -u
sfpi_version=7.1.0-ext-29186
sfpi_x86_64_deb_md5=649ab243ae65ffeab0d7b7ebba622f2a
sfpi_x86_64_rpm_md5=4026eff2cce55d81da4948d76c0ae4f0
sfpi_x86_64_txz_md5=e32fcc3b5cb100509751a692dbd85461
