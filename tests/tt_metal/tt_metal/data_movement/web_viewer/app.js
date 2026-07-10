// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// ── Configuration ──────────────────────────────────────────────────

const DATA_BASE_PATH = "../data";
const KNOWN_ARCHITECTURES = ["blackhole", "wormhole_b0"];
const GITHUB_BASE =
    "https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/data_movement";

const TEST_GROUPS = [
    {
        group: "One from One",
        directory: "one_from_one",
        tags: ["read", "unicast", "L1", "single_core"],
        tests: [
            { name: "Packet Sizes", csv: "One from One Packet Sizes 2.0.csv" },
            { name: "Virtual Channels", csv: "One from One Virtual Channels.csv" },
        ],
    },
    {
        group: "One to One",
        directory: "one_to_one",
        tags: ["write", "unicast", "L1", "single_core"],
        tests: [
            { name: "Packet Sizes", csv: "One to One Packet Sizes 2.0.csv" },
            { name: "Virtual Channels", csv: "One to One Virtual Channels.csv" },
        ],
    },
    {
        group: "One from All",
        directory: "one_from_all",
        tags: ["read", "unicast", "L1", "multi_core"],
        tests: [
            { name: "Packet Sizes", csv: "One from All Packet Sizes 2.0.csv" },
            { name: "Virtual Channels", csv: "One from All Virtual Channels 2.0.csv" },
        ],
    },
    {
        group: "One to All",
        directory: "one_to_all",
        tags: ["write", "unicast", "multicast", "L1", "multi_core"],
        tests: [
            { name: "Unicast Packet Sizes", csv: "One to All Unicast Packet Sizes 2.0.csv" },
            { name: "Multicast Packet Sizes", csv: "One to All Multicast Packet Sizes 2.0.csv" },
            { name: "Multicast Linked Packet Sizes", csv: "One to All Multicast Linked Packet Sizes 2.0.csv" },
            { name: "Multicast Schemes (Loopback Disabled)", csv: "Multicast Schemes (Loopback Disabled).csv" },
        ],
    },
    {
        group: "One Packet",
        directory: "one_packet",
        tags: ["read", "write", "unicast", "L1", "single_core"],
        tests: [
            { name: "Read Sizes", csv: "One Packet Read Sizes 2.0.csv" },
            { name: "Write Sizes", csv: "One Packet Write Sizes 2.0.csv" },
        ],
    },
    {
        group: "All to All",
        directory: "all_to_all",
        tags: ["write", "unicast", "L1", "multi_core"],
        tests: [
            { name: "Packet Sizes", csv: "All to All Packet Sizes 2.0.csv" },
            { name: "Grid Sweep Packet Sizes", csv: "All to All Grid Sweep Packet Sizes 2.0.csv" },
        ],
    },
    {
        group: "All from All",
        directory: "all_from_all",
        tags: ["read", "unicast", "L1", "multi_core"],
        tests: [
            { name: "Packet Sizes", csv: "All from All Packet Sizes.csv" },
            { name: "Grid Sweep Packet Sizes", csv: "All from All Grid Sweep Packet Sizes 2.0.csv" },
        ],
    },
    {
        group: "Multi Interleaved",
        directory: "multi_interleaved",
        tags: ["read", "write", "unicast", "DRAM", "multi_core"],
        tests: [
            { name: "Sizes", csv: "Multi Interleaved Sizes.csv" },
            { name: "Read Sizes", csv: "Multi Interleaved Read Sizes.csv" },
            { name: "Write Sizes", csv: "Multi Interleaved Write Sizes.csv" },
            { name: "Grid Sweep Sizes", csv: "Multi Interleaved Grid Sweep Sizes.csv" },
            { name: "Grid Sweep Read Sizes", csv: "Multi Interleaved Read Grid Sweep Sizes.csv" },
            { name: "Grid Sweep Write Sizes", csv: "Multi Interleaved Write Grid Sweep Sizes.csv" },
        ],
    },
    {
        group: "Atomic Semaphore",
        directory: "atomics",
        tags: ["write", "unicast", "L1", "multi_core", "atomics"],
        tests: [
            { name: "Adjacent Bandwidth Sweep", csv: "Atomic Semaphore Adjacent Bandwidth Sweep.csv" },
            { name: "Non Adjacent Bandwidth Sweep", csv: "Atomic Semaphore Non Adjacent Bandwidth Sweep.csv" },
        ],
    },
    {
        group: "Core Bidirectional",
        directory: "core_bidirectional",
        tags: ["read", "write", "unicast", "L1", "single_core"],
        tests: [
            { name: "Packet Sizes (Same Kernel)", csv: "Core Bidirectional Packet Sizes Same Kernel.csv" },
            { name: "Packet Sizes (Different Kernels)", csv: "Core Bidirectional Packet Sizes Different Kernels.csv" },
        ],
    },
    {
        group: "Direct Write",
        directory: "direct_write",
        tags: ["write", "unicast", "L1", "single_core"],
        tests: [
            { name: "Performance Comparison", csv: "Inline Direct Write Performance Comparison.csv" },
            { name: "Address Pattern", csv: "Inline Direct Write Address Pattern.csv" },
        ],
    },
    {
        group: "DRAM Neighbour",
        directory: "dram_neighbour",
        tags: ["read", "unicast", "DRAM", "multi_core"],
        tests: [
            { name: "Pages Sweep (Closest)", csv: "Number of Pages Sweep for DRAM Closest Neighbour.csv" },
            { name: "Banks Sweep (Closest)", csv: "Number of Banks Sweep for DRAM Closest Neighbour.csv" },
            { name: "Pages Sweep (Single Row)", csv: "Number of Pages Sweep for Single Row DRAM.csv" },
            { name: "Pages Sweep (One Hop)", csv: "Number of Pages Sweep for DRAM One Hop.csv" },
            { name: "Pages Sweep (Loop Back)", csv: "Number of Pages Sweep for DRAM Loop Back.csv" },
        ],
    },
    {
        group: "DRAM Sharded Read",
        directory: "dram_sharded",
        tags: ["read", "unicast", "DRAM", "multi_core"],
        tests: [
            { name: "Tile Numbers", csv: "DRAM Sharded Read Tile Numbers 2.0.csv" },
            { name: "Bank Numbers", csv: "DRAM Sharded Read Bank Numbers 2.0.csv" },
        ],
    },
    {
        group: "DRAM Unary",
        directory: "dram_unary",
        tags: ["read", "write", "unicast", "DRAM", "single_core"],
        tests: [
            { name: "Packet Sizes", csv: "DRAM Packet Sizes 2.0.csv" },
            { name: "Channels", csv: "DRAM Channels 2.0.csv" },
            { name: "Core Locations", csv: "DRAM Core Locations 2.0.csv" },
        ],
    },
    {
        group: "DRAM Interleaved",
        directory: "interleaved",
        tags: ["read", "write", "unicast", "DRAM", "multi_core"],
        tests: [
            { name: "Page Numbers", csv: "DRAM Interleaved Page Numbers.csv" },
            { name: "Read Numbers", csv: "DRAM Interleaved Page Read Numbers.csv" },
            { name: "Write Numbers", csv: "DRAM Interleaved Page Write Numbers.csv" },
        ],
    },
    {
        group: "L1 Interleaved",
        directory: "interleaved",
        tags: ["read", "write", "unicast", "L1", "multi_core"],
        tests: [
            { name: "Page Numbers", csv: "L1 Interleaved Page Numbers.csv" },
            { name: "Read Numbers", csv: "L1 Interleaved Page Read Numbers.csv" },
            { name: "Write Numbers", csv: "L1 Interleaved Page Write Numbers.csv" },
        ],
    },
    {
        group: "Loopback",
        directory: "loopback",
        tags: ["read", "write", "unicast", "L1", "single_core"],
        tests: [
            { name: "Packet Sizes", csv: "Loopback Packet Sizes.csv" },
        ],
    },
    {
        group: "Noc Estimator",
        directory: "noc_estimator_tests",
        tags: ["read", "write", "unicast", "multicast", "L1", "DRAM", "multi_core"],
        tests: [
            { name: "L1 One to One", csv: "Noc Estimator - L1 One to One.csv" },
            { name: "L1 One from One", csv: "Noc Estimator - L1 One from One.csv" },
            { name: "L1 One to All", csv: "Noc Estimator - L1 One to All.csv" },
            { name: "L1 One from All", csv: "Noc Estimator - L1 One from All.csv" },
            { name: "L1 All to All", csv: "Noc Estimator - L1 All to All.csv" },
            { name: "L1 All from All", csv: "Noc Estimator - L1 All from All.csv" },
            { name: "L1 One to Row", csv: "Noc Estimator - L1 One to Row.csv" },
            { name: "L1 Row to Row", csv: "Noc Estimator - L1 Row to Row.csv" },
            { name: "L1 One to Column", csv: "Noc Estimator - L1 One to Column.csv" },
            { name: "L1 Column to Column", csv: "Noc Estimator - L1 Column to Column.csv" },
            { name: "DRAM Sharded One from One", csv: "Noc Estimator - DRAM Sharded One from One.csv" },
            { name: "DRAM Sharded All from All", csv: "Noc Estimator - DRAM Sharded All from All.csv" },
            { name: "DRAM Interleaved One from All", csv: "Noc Estimator - DRAM Interleaved One from All.csv" },
            { name: "DRAM Interleaved All from All", csv: "Noc Estimator - DRAM Interleaved All from All.csv" },
            { name: "DRAM Sharded One to One", csv: "Noc Estimator - DRAM Sharded One to One.csv" },
            { name: "DRAM Sharded All to All", csv: "Noc Estimator - DRAM Sharded All to All.csv" },
            { name: "DRAM Interleaved One to All", csv: "Noc Estimator - DRAM Interleaved One to All.csv" },
            { name: "DRAM Interleaved All to All", csv: "Noc Estimator - DRAM Interleaved All to All.csv" },
        ],
    },
    {
        group: "PCIe Read",
        directory: "pcie_read_bw",
        tags: ["read", "unicast", "DRAM", "single_core"],
        tests: [
            { name: "Bandwidth Sweep", csv: "PCIe Read Bandwidth Sweep.csv" },
        ],
    },
    {
        group: "PCIe Write",
        directory: "pcie_write_bw",
        tags: ["write", "unicast", "DRAM", "single_core"],
        tests: [
            { name: "Bandwidth Sweep", csv: "PCIe Write Bandwidth Sweep.csv" },
        ],
    },
    {
        group: "Transaction ID",
        directory: "transaction_id",
        tags: ["read", "write", "unicast", "L1", "single_core"],
        tests: [
            { name: "Read After Write", csv: "Transaction ID - Read After Write 2.0.csv" },
            { name: "Read After Write (One Packet)", csv: "Transaction ID - Read After Write One Packet 2.0.csv" },
            { name: "Read After Write (One Packet Stateful)", csv: "Transaction ID - Read After Write One Packet Stateful 2.0.csv" },
            { name: "Write After Read", csv: "Transaction ID - Write After Read 2.0.csv" },
            { name: "Write After Read (One Packet Stateful)", csv: "Transaction ID - Write After Read One Packet Stateful 2.0.csv" },
        ],
    },
];

const INTERNAL_COLUMNS = new Set([
    "Run Host ID",
    "Architecture",
    "Grid Size X",
    "Grid Size Y",
    "Subordinate Grid Size X",
    "Subordinate Grid Size Y",
    "Master Grid Size X",
    "Master Grid Size Y",
]);

const X_AXIS_EXCLUDED = new Set([
    "NoC Index",
]);

const METRIC_COLUMNS = new Set([
    "Latency (cycles)",
    "Bandwidth (bytes/cycle)",
    "Combined Bandwidth (bytes/cycle)",
    "Bandwidth (GB/s)",
    "Total Bytes",
    "Number of Cores",
    "Clock Frequency (MHz)",
]);

const X_AXIS_PRIORITY = [
    "Transaction Size (bytes)",
    "Grid Dimensions",
    "Master Grid Dimensions",
    "Subordinate Grid Dimensions",
];

const SERIES_COLORS = [
    "#7b68ee",
    "#ff6b6b",
    "#4ecdc4",
    "#ffa726",
    "#66bb6a",
    "#ab47bc",
    "#29b6f6",
    "#ef5350",
    "#26a69a",
    "#ffc107",
];

// ── State ──────────────────────────────────────────────────────────

let state = {
    selectedGroup: null,
    selectedTest: null,
    selectedArch: null,
    availableArchs: [],
    activeTags: new Set(),
    searchQuery: "",
    yAxisMode: "bandwidth",
    visibleValues: new Map(),
    csvData: null,
    analysis: null,
    chartConfig: null,
};

// ── Column Analysis ────────────────────────────────────────────────

function analyzeColumns(rows) {
    if (!rows.length) return null;

    const allColumns = Object.keys(rows[0]);
    const metrics = [];
    const sweepDimensions = [];
    const constants = {};

    for (const col of allColumns) {
        if (INTERNAL_COLUMNS.has(col) || METRIC_COLUMNS.has(col)) {
            if (METRIC_COLUMNS.has(col)) metrics.push(col);
            continue;
        }

        const values = [...new Set(rows.map((r) => r[col]))];
        if (values.length <= 1) {
            constants[col] = values[0];
        } else {
            sweepDimensions.push({ column: col, values });
        }
    }

    return { metrics, sweepDimensions, constants };
}

function buildChartConfig(analysis, chosenXAxis) {
    const { metrics, sweepDimensions } = analysis;

    const bandwidthCol =
        metrics.find((m) => m.includes("Bandwidth")) || null;
    const latencyCol = metrics.find((m) => m.includes("Latency")) || null;

    const allDimensions = sweepDimensions
        .map((d) => d.column)
        .filter((d) => !X_AXIS_EXCLUDED.has(d));

    let xAxis;
    if (chosenXAxis && allDimensions.includes(chosenXAxis)) {
        xAxis = chosenXAxis;
    } else {
        xAxis = X_AXIS_PRIORITY.find((d) => allDimensions.includes(d))
            || allDimensions[0];
    }

    const groupers = sweepDimensions
        .filter((d) => d.column !== xAxis)
        .map((d) => d.column);

    return { xAxis, groupers, allDimensions, bandwidthCol, latencyCol };
}

// ── CSV Loading ────────────────────────────────────────────────────

function loadCSV(url) {
    return new Promise((resolve, reject) => {
        Papa.parse(url, {
            download: true,
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => resolve(results.data),
            error: (err) => reject(err),
        });
    });
}

async function probeArchitectures(test) {
    const available = [];
    const probes = KNOWN_ARCHITECTURES.map(async (arch) => {
        const url = `${DATA_BASE_PATH}/${arch}/${test.csv}`;
        try {
            const resp = await fetch(url);
            if (resp.ok) available.push(arch);
        } catch {
            // not available
        }
    });
    await Promise.all(probes);
    available.sort();
    return available;
}

// ── Tag Collection ─────────────────────────────────────────────────

function getAllTags() {
    const tags = new Set();
    for (const group of TEST_GROUPS) {
        for (const tag of group.tags) {
            tags.add(tag);
        }
    }
    return [...tags].sort();
}

// ── Filtering ──────────────────────────────────────────────────────

function getFilteredGroups() {
    const query = state.searchQuery.toLowerCase();
    return TEST_GROUPS.filter((group) => {
        if (query) {
            const groupMatch = group.group.toLowerCase().includes(query);
            const testMatch = group.tests.some((t) =>
                t.name.toLowerCase().includes(query)
            );
            if (!groupMatch && !testMatch) return false;
        }
        if (state.activeTags.size > 0) {
            for (const tag of state.activeTags) {
                if (!group.tags.includes(tag)) return false;
            }
        }
        return true;
    });
}

// ── Chart Rendering (Plotly) ───────────────────────────────────────

function parseGridDimension(val) {
    const m = String(val).match(/^(\d+)\s*x\s*(\d+)$/);
    if (!m) return null;
    return { w: Number(m[1]), h: Number(m[2]), total: Number(m[1]) * Number(m[2]) };
}

function compareAxisValues(a, b) {
    const ga = parseGridDimension(a);
    const gb = parseGridDimension(b);
    if (ga && gb) {
        if (ga.total !== gb.total) return ga.total - gb.total;
        if (ga.w !== gb.w) return ga.w - gb.w;
        return ga.h - gb.h;
    }
    const na = Number(a), nb = Number(b);
    if (!isNaN(na) && !isNaN(nb)) return na - nb;
    return String(a).localeCompare(String(b));
}

function formatBytes(value) {
    if (value >= 1024) {
        return value / 1024 + "K";
    }
    return String(value);
}

function getPlotData(rows, config) {
    const yCol =
        state.yAxisMode === "bandwidth"
            ? config.bandwidthCol
            : config.latencyCol;

    if (!yCol) return new Map();

    const series = new Map();
    for (const row of rows) {
        let visible = true;
        for (const dim of config.groupers) {
            const allowed = state.visibleValues.get(dim);
            if (allowed && !allowed.has(String(row[dim]))) {
                visible = false;
                break;
            }
        }
        if (!visible) continue;

        const seriesKey = config.groupers.length > 0
            ? config.groupers.map((d) => String(row[d])).join(" | ")
            : "all";

        if (!series.has(seriesKey)) {
            series.set(seriesKey, { x: [], y: [] });
        }
        const s = series.get(seriesKey);
        s.x.push(row[config.xAxis]);
        s.y.push(row[yCol]);
    }

    for (const s of series.values()) {
        const indices = s.x.map((_, i) => i).sort((a, b) => compareAxisValues(s.x[a], s.x[b]));
        s.x = indices.map((i) => s.x[i]);
        s.y = indices.map((i) => s.y[i]);
    }

    return series;
}

function renderChart(rows, config) {
    const plotData = getPlotData(rows, config);

    const traces = [];
    let colorIdx = 0;
    const sortedKeys = [...plotData.keys()].sort((a, b) => {
        const aParts = a.split(" | ");
        const bParts = b.split(" | ");
        for (let i = 0; i < aParts.length; i++) {
            const na = Number(aParts[i]);
            const nb = Number(bParts[i]);
            if (!isNaN(na) && !isNaN(nb)) {
                if (na !== nb) return na - nb;
            } else {
                const cmp = (aParts[i] || "").localeCompare(bParts[i] || "");
                if (cmp !== 0) return cmp;
            }
        }
        return 0;
    });

    for (const key of sortedKeys) {
        const s = plotData.get(key);
        const color = SERIES_COLORS[colorIdx % SERIES_COLORS.length];
        traces.push({
            x: s.x,
            y: s.y,
            type: "scatter",
            mode: "lines+markers",
            name: config.groupers.length > 0 ? key : "Data",
            line: { color, width: 2 },
            marker: { color, size: 5 },
            hovertemplate:
                `<b>${config.xAxis}:</b> %{x}<br>` +
                `<b>%{fullData.name}:</b> %{y:.4f}<extra></extra>`,
        });
        colorIdx++;
    }

    const isLatency = state.yAxisMode === "latency";
    const yLabel = isLatency
        ? config.latencyCol || "Latency (cycles)"
        : config.bandwidthCol || "Bandwidth (bytes/cycle)";

    const xIsNumeric = traces.length > 0 &&
        traces[0].x.length > 0 &&
        traces.every((t) => t.x.every((v) => typeof v === "number" && isFinite(v) && v > 0));

    const xaxis = {
        title: { text: config.xAxis, font: { color: "#a0a0b0" } },
        tickfont: { color: "#a0a0b0" },
        gridcolor: "#2a2a4a",
        zerolinecolor: "#2a2a4a",
        linecolor: "#2a2a4a",
    };

    if (xIsNumeric) {
        xaxis.type = "log";
        xaxis.dtick = Math.log10(2);
        xaxis.tickformat = "";
        xaxis.tickvals = getBase2Ticks(traces);
        xaxis.ticktext = getBase2Ticks(traces).map(formatBytes);
    } else {
        xaxis.type = "category";
        const allCatValues = [...new Set(traces.flatMap((t) => t.x))];
        allCatValues.sort(compareAxisValues);
        xaxis.categoryorder = "array";
        xaxis.categoryarray = allCatValues;
    }

    const layout = {
        xaxis,
        yaxis: {
            type: isLatency ? "log" : "linear",
            title: { text: yLabel, font: { color: "#a0a0b0" } },
            tickfont: { color: "#a0a0b0" },
            gridcolor: "#2a2a4a",
            zerolinecolor: "#2a2a4a",
            linecolor: "#2a2a4a",
            rangemode: isLatency ? "normal" : "tozero",
        },
        plot_bgcolor: "#16213e",
        paper_bgcolor: "#16213e",
        font: { color: "#e0e0e0" },
        legend: {
            font: { color: "#e0e0e0" },
            bgcolor: "rgba(0,0,0,0)",
        },
        margin: { t: 30, r: 30, b: 60, l: 70 },
        hovermode: "x unified",
    };

    const plotConfig = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
        displaylogo: false,
    };

    Plotly.newPlot("chart", traces, layout, plotConfig);
}

function getBase2Ticks(traces) {
    let min = Infinity;
    let max = -Infinity;
    for (const t of traces) {
        for (const v of t.x) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
    }
    if (!isFinite(min) || min <= 0) return [];
    const ticks = [];
    let exp = Math.floor(Math.log2(min));
    while (Math.pow(2, exp) <= max) {
        const val = Math.pow(2, exp);
        if (val >= min) ticks.push(val);
        exp++;
    }
    return ticks;
}

// ── UI Rendering ───────────────────────────────────────────────────

function renderTagChips() {
    const container = document.getElementById("tag-chips");
    container.innerHTML = "";
    for (const tag of getAllTags()) {
        const chip = document.createElement("span");
        chip.className =
            "tag-chip" + (state.activeTags.has(tag) ? " active" : "");
        chip.textContent = tag;
        chip.addEventListener("click", () => {
            if (state.activeTags.has(tag)) {
                state.activeTags.delete(tag);
            } else {
                state.activeTags.add(tag);
            }
            renderTagChips();
            renderTestList();
        });
        container.appendChild(chip);
    }
}

function renderTestList() {
    const list = document.getElementById("test-list");
    list.innerHTML = "";
    const filtered = getFilteredGroups();

    for (const group of filtered) {
        const isExpanded = state.selectedGroup === group;

        const li = document.createElement("li");
        li.className = "group-item" + (isExpanded ? " expanded" : "");

        const header = document.createElement("div");
        header.className = "group-header";
        header.textContent = group.group;
        header.addEventListener("click", () => {
            state.selectedGroup = isExpanded ? null : group;
            if (!isExpanded && group.tests.length === 1) {
                selectTest(group, group.tests[0]);
            } else {
                renderTestList();
            }
        });
        li.appendChild(header);

        const tagsDiv = document.createElement("div");
        tagsDiv.className = "test-tags";
        for (const tag of group.tags) {
            const tagSpan = document.createElement("span");
            tagSpan.className = "test-tag";
            tagSpan.textContent = tag;
            tagsDiv.appendChild(tagSpan);
        }
        li.appendChild(tagsDiv);

        if (isExpanded && group.tests.length > 1) {
            const subList = document.createElement("ul");
            subList.className = "sub-test-list";
            for (const test of group.tests) {
                const subLi = document.createElement("li");
                subLi.className = state.selectedTest === test ? "selected" : "";
                subLi.textContent = test.name;
                subLi.addEventListener("click", (e) => {
                    e.stopPropagation();
                    selectTest(group, test);
                });
                subList.appendChild(subLi);
            }
            li.appendChild(subList);
        }

        list.appendChild(li);
    }
}

function renderArchChips() {
    const container = document.getElementById("arch-chips");
    container.innerHTML = "";

    for (const arch of state.availableArchs) {
        const chip = document.createElement("span");
        chip.className =
            "arch-chip" + (state.selectedArch === arch ? " active" : "");
        chip.textContent = arch;
        chip.addEventListener("click", () => {
            if (state.selectedArch !== arch) {
                state.selectedArch = arch;
                loadAndRender();
            }
        });
        container.appendChild(chip);
    }
}

function renderTestInfo(analysis) {
    const container = document.getElementById("test-info");
    container.innerHTML = "";

    if (!analysis) return;

    for (const [key, value] of Object.entries(analysis.constants)) {
        const item = document.createElement("span");
        item.className = "info-item";
        item.innerHTML = `<span class="info-label">${key}:</span> <span class="info-value">${value}</span>`;
        container.appendChild(item);
    }
}

function renderXAxisSelector(config) {
    const container = document.getElementById("x-axis-selector");
    const select = document.getElementById("x-axis-select");

    if (config.allDimensions.length <= 1) {
        container.style.display = "none";
        return;
    }

    container.style.display = "";
    select.innerHTML = "";

    for (const dim of config.allDimensions) {
        const option = document.createElement("option");
        option.value = dim;
        option.textContent = dim;
        option.selected = dim === config.xAxis;
        select.appendChild(option);
    }

    select.onchange = () => {
        const newConfig = buildChartConfig(state.analysis, select.value);
        state.chartConfig = newConfig;
        renderSweepFilters(newConfig, state.csvData);
        renderChart(state.csvData, newConfig);
    };
}

function renderSweepFilters(config, rows) {
    const container = document.getElementById("sweep-filters");
    container.innerHTML = "";

    state.visibleValues = new Map();

    for (const dim of config.groupers) {
        const values = [...new Set(rows.map((r) => r[dim]))]
            .sort(compareAxisValues);

        state.visibleValues.set(dim, new Set(values.map(String)));

        const group = document.createElement("div");
        group.className = "series-controls";

        const label = document.createElement("span");
        label.className = "label";
        label.textContent = dim + ":";
        group.appendChild(label);

        const checkboxContainer = document.createElement("div");
        checkboxContainer.className = "series-checkboxes";

        values.forEach((val) => {
            const lbl = document.createElement("label");
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.checked = true;
            checkbox.addEventListener("change", () => {
                const key = String(val);
                const visible = state.visibleValues.get(dim);
                if (checkbox.checked) {
                    visible.add(key);
                } else {
                    visible.delete(key);
                }
                renderChart(state.csvData, state.chartConfig);
            });

            lbl.appendChild(checkbox);
            lbl.appendChild(document.createTextNode(" " + val));
            checkboxContainer.appendChild(lbl);
        });

        group.appendChild(checkboxContainer);
        container.appendChild(group);
    }
}

// ── Main Actions ───────────────────────────────────────────────────

async function selectTest(group, test) {
    state.selectedGroup = group;
    state.selectedTest = test;

    document.getElementById("empty-state").style.display = "none";
    document.getElementById("test-view").style.display = "block";

    document.getElementById("test-name").textContent =
        group.group + " — " + test.name;
    document.getElementById("readme-link").href =
        `${GITHUB_BASE}/${group.directory}/README.md`;

    renderTestList();

    state.availableArchs = await probeArchitectures(test);
    if (state.availableArchs.length === 0) {
        document.getElementById("test-info").innerHTML =
            '<span class="info-item"><span class="info-value">No CSV data found for any architecture.</span></span>';
        document.getElementById("arch-chips").innerHTML = "";
        document.getElementById("x-axis-selector").style.display = "none";
        document.getElementById("sweep-filters").innerHTML = "";
        Plotly.purge("chart");
        return;
    }

    if (
        !state.selectedArch ||
        !state.availableArchs.includes(state.selectedArch)
    ) {
        state.selectedArch = state.availableArchs[0];
    }
    renderArchChips();
    await loadAndRender();
}

async function loadAndRender() {
    const test = state.selectedTest;
    if (!test || !state.selectedArch) return;

    const url = `${DATA_BASE_PATH}/${state.selectedArch}/${test.csv}`;

    try {
        const rows = await loadCSV(url);
        state.csvData = rows;

        const analysis = analyzeColumns(rows);
        state.analysis = analysis;
        const config = buildChartConfig(analysis);
        state.chartConfig = config;

        renderArchChips();
        renderTestInfo(analysis);
        renderXAxisSelector(config);
        renderSweepFilters(config, rows);
        renderChart(rows, config);

        setupYAxisListeners();
    } catch (err) {
        document.getElementById("test-info").innerHTML =
            `<span class="info-item"><span class="info-value">Error loading CSV: ${err.message}</span></span>`;
    }
}

function setupYAxisListeners() {
    const radios = document.querySelectorAll('input[name="y-axis"]');
    const handler = (e) => {
        state.yAxisMode = e.target.value;
        renderChart(state.csvData, state.chartConfig);
    };
    for (const radio of radios) {
        const newRadio = radio.cloneNode(true);
        newRadio.checked = radio.value === state.yAxisMode;
        radio.parentNode.replaceChild(newRadio, radio);
        newRadio.addEventListener("change", handler);
    }
}

// ── Init ───────────────────────────────────────────────────────────

function init() {
    renderTagChips();
    renderTestList();

    document.getElementById("search-input").addEventListener("input", (e) => {
        state.searchQuery = e.target.value;
        renderTestList();
    });
}

document.addEventListener("DOMContentLoaded", init);
