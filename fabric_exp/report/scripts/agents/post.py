import json, sys
D = "/home/aagarwal/tt-metal/fabric_exp/report/data/"
for f in ("fanin", "stability"):
    r = json.load(open(D + f + ".json"))
    for x in r:
        ol = x.get("opt_level")
        if ol is not None and ol not in ("-O3", "-Os", "-O2", "auto"):
            if ol.startswith("-Os"):
                x["opt_level"] = "-Os"
            else:
                del x["opt_level"]  # comparisons across levels: the levels are in config
        if x["id"].startswith("p150-census-mux-") and not x["id"].startswith("p150-census-mux-mean-v4rx"):
            x["keepalive"] = "firmware (8000 cycles)" if x["router"] == "main" else "1000 cycles (study setting)"
        if x["metric"] == "rate_suite_mean":
            if x["id"].startswith("p150-census-mux-mean-") and x["router"] in ("main", "v4trm"):
                x["test"] = "SUITE MEAN (22 rows)"
            elif x["id"].startswith("p150-census-mux-mean-"):
                x["test"] = "SUITE MEAN (mux suite; row count not restated in §17)"
            elif x["id"].startswith("opt-367-vc1i"):
                x["test"] = "SUITE MEAN (8 rows, vc1_inter)"
            elif x["id"].startswith("opt-367-vc1"):
                x["test"] = "SUITE MEAN (13 rows, vc1_alloc; MixCmd excluded)"
    json.dump(r, open(D + f + ".json", "w"), indent=1)
    print(f, len(r))
