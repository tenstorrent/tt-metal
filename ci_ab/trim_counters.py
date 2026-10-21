#!/usr/bin/env python3
"""Shrink counter CSVs in place: drop columns that are zero in every row. usage: trim_counters.py <csv dir>"""
import glob, sys
import pandas as pd

for f in glob.glob(f"{sys.argv[1]}/*/*.csv"):
    if "counter" not in f:
        continue
    try:
        x = pd.read_csv(f, low_memory=False)
    except Exception as e:  # keep the file as it is
        print("trim: skip", f, e)
        continue
    num = x.select_dtypes("number")
    zero = [c for c in num.columns if (num[c] == 0).all()]
    x.drop(columns=zero).to_csv(f, index=False)
