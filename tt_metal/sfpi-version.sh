# set SFPI release version information

sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi_\([-.0-9a-zA-Z]*\)_\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/'\
# 'sfpi_version=\2'"\n"'sfpi_\3_\4_md5=\1/' *-build/sfpi_*.md5 | sort -u
sfpi_version=7.3.0-replay-29538
sfpi_x86_64_deb_md5=b0d8e22076822f52b6a393af5d560385
sfpi_x86_64_rpm_md5=4823c0f49377329940865cf811f9051d
sfpi_x86_64_txz_md5=cf25dfc7b0e3f04c2a0522ac8c04ba8f
