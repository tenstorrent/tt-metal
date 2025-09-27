# set SFPI release version information

sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi_\([-.0-9a-zA-Z]*\)_\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_version=\2'"\n"'sfpi_\3_\4_md5=\1/' *-build/sfpi_*.md5 | sort -u
sfpi_aarch64_deb_md5=a67c257092b55af74b16d1ca03305f05
sfpi_aarch64_rpm_md5=3ec63a0a7b83485867f103c7fd3f5079
sfpi_aarch64_txz_md5=6f0380cb9d50055e0c31f5b7e96fb638
sfpi_version=7.2.0
sfpi_x86_64_deb_md5=df7a206658fc40cf31fe7282343a2752
sfpi_x86_64_rpm_md5=9634997e6101227dae10bd9701f046e8
sfpi_x86_64_txz_md5=e408d57cc776d4d51055323c5e29cf02
