# set SFPI release version information

sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi_\([-.0-9a-zA-Z]*\)_\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_version=\2'"\n"'sfpi_\3_\4_md5=\1/' *-build/sfpi_*.md5 | sort -u
sfpi_version=7.1.0-ttinsn2
sfpi_x86_64_deb_md5=e7efb28c87d27edd21cb6be911470c27
sfpi_x86_64_rpm_md5=9a4e6e39aad883596886c8898a4e237e
sfpi_x86_64_txz_md5=8b37efaa34b048730173906dc034d7df
