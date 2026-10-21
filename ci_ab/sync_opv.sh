#!/bin/bash
# opv/H = the base tree's files, opv/S = the pair path's (wtsp working tree)
R=/localdev/mvlahovic/llk_analysis_builds/redall; D=$R/ciwt_sp/ci_ab/opv
while read -r rel; do
  mkdir -p $D/H/$(dirname $rel) $D/S/$(dirname $rel)
  cp $R/sides/base/$rel $D/H/$rel; cp $R/wtsp/$rel $D/S/$rel
done < $R/ciwt_sp/ci_ab/opl/files_sp.txt
