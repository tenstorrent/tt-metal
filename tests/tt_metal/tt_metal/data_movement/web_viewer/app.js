// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// ── Configuration ──────────────────────────────────────────────────

const DATA_BASE_PATH = "../data";
const KNOWN_ARCHITECTURES = ["blackhole", "wormhole_b0"];
const GITHUB_BASE =
    "https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/data_movement";

const TESTS = [
    // ── One from One ───────────────────────────────────────────────
    {
        name: "One from One Packet Sizes",
        csv: "One from One Packet Sizes.csv",
        tags: ["read", "unicast", "L1", "single_core"],
        directory: "one_from_one",
    },
    {
        name: "One from One Packet Sizes 2.0",
        csv: "One from One Packet Sizes 2.0.csv",
        tags: ["read", "unicast", "L1", "single_core", "2.0_api"],
        directory: "one_from_one",
    },
    {
        name: "One from One Directed Ideal",
        csv: "One from One Directed Ideal.csv",
        tags: ["read", "unicast", "L1", "single_core", "directed_ideal"],
        directory: "one_from_one",
    },
    {
        name: "One from One Virtual Channels",
        csv: "One from One Virtual Channels.csv",
        tags: ["read", "unicast", "L1", "single_core", "virtual_channels"],
        directory: "one_from_one",
    },

    // ── One to One ─────────────────────────────────────────────────
    {
        name: "One to One Packet Sizes",
        csv: "One to One Packet Sizes.csv",
        tags: ["write", "unicast", "L1", "single_core"],
        directory: "one_to_one",
    },
    {
        name: "One to One Packet Sizes 2.0",
        csv: "One to One Packet Sizes 2.0.csv",
        tags: ["write", "unicast", "L1", "single_core", "2.0_api"],
        directory: "one_to_one",
    },
    {
        name: "One to One Directed Ideal",
        csv: "One to One Directed Ideal.csv",
        tags: ["write", "unicast", "L1", "single_core", "directed_ideal"],
        directory: "one_to_one",
    },
    {
        name: "One to One Virtual Channels",
        csv: "One to One Virtual Channels.csv",
        tags: ["write", "unicast", "L1", "single_core", "virtual_channels"],
        directory: "one_to_one",
    },

    // ── One from All ───────────────────────────────────────────────
    {
        name: "One from All Packet Sizes",
        csv: "One from All Packet Sizes.csv",
        tags: ["read", "unicast", "L1", "multi_core"],
        directory: "one_from_all",
    },
    {
        name: "One from All Directed Ideal",
        csv: "One from All Directed Ideal.csv",
        tags: ["read", "unicast", "L1", "multi_core", "directed_ideal"],
        directory: "one_from_all",
    },
    {
        name: "One from All Virtual Channels",
        csv: "One from All Virtual Channels.csv",
        tags: ["read", "unicast", "L1", "multi_core", "virtual_channels"],
        directory: "one_from_all",
    },

    // ── One to All ─────────────────────────────────────────────────
    {
        name: "One to All Multicast Linked Packet Sizes",
        csv: "One to All Multicast Linked Packet Sizes.csv",
        tags: ["write", "multicast", "L1", "multi_core"],
        directory: "one_to_all",
    },
    {
        name: "One to All Multicast Linked Packet Sizes 2.0",
        csv: "One to All Multicast Linked Packet Sizes 2.0.csv",
        tags: ["write", "multicast", "L1", "multi_core", "2.0_api"],
        directory: "one_to_all",
    },
    {
        name: "Multicast Schemes (Loopback Disabled)",
        csv: "Multicast Schemes (Loopback Disabled).csv",
        tags: ["write", "multicast", "L1", "multi_core", "2.0_api"],
        directory: "one_to_all",
    },

    // ── All to All ─────────────────────────────────────────────────
    {
        name: "All to All Packet Sizes 2.0",
        csv: "All to All Packet Sizes 2.0.csv",
        tags: ["write", "unicast", "L1", "multi_core", "2.0_api"],
        directory: "all_to_all",
    },

    // ── All from All ───────────────────────────────────────────────
    {
        name: "All from All Packet Sizes",
        csv: "All from All Packet Sizes.csv",
        tags: ["read", "unicast", "L1", "multi_core"],
        directory: "all_from_all",
    },
    {
        name: "All from All Directed Ideal",
        csv: "All from All Directed Ideal.csv",
        tags: ["read", "unicast", "L1", "multi_core", "directed_ideal"],
        directory: "all_from_all",
    },
    {
        name: "All from All 2x2 From 1x1 Directed Ideal",
        csv: "All from All 2x2 From 1x1 Directed Ideal.csv",
        tags: ["read", "unicast", "L1", "multi_core", "directed_ideal"],
        directory: "all_from_all",
    },
    {
        name: "All from All 4x4 From 1x1 Directed Ideal",
        csv: "All from All 4x4 From 1x1 Directed Ideal.csv",
        tags: ["read", "unicast", "L1", "multi_core", "directed_ideal"],
        directory: "all_from_all",
    },
    {
        name: "All from All 1x1 From 2x2 Directed Ideal",
        csv: "All from All 1x1 From 2x2 Directed Ideal.csv",
        tags: ["read", "unicast", "L1", "multi_core", "directed_ideal"],
        directory: "all_from_all",
    },
    {
        name: "All from All 1x1 From 4x4 Directed Ideal",
        csv: "All from All 1x1 From 4x4 Directed Ideal.csv",
        tags: ["read", "unicast", "L1", "multi_core", "directed_ideal"],
        directory: "all_from_all",
    },
    {
        name: "All from All 2x2 From 2x2 Directed Ideal",
        csv: "All from All 2x2 From 2x2 Directed Ideal.csv",
        tags: ["read", "unicast", "L1", "multi_core", "directed_ideal"],
        directory: "all_from_all",
    },

    // ── Multi Interleaved ──────────────────────────────────────────
    {
        name: "Multi Interleaved Sizes",
        csv: "Multi Interleaved Sizes.csv",
        tags: ["read", "unicast", "DRAM", "multi_core"],
        directory: "multi_interleaved",
    },

    // ── Atomics ────────────────────────────────────────────────────
    {
        name: "Atomic Semaphore Adjacent Bandwidth Sweep",
        csv: "Atomic Semaphore Adjacent Bandwidth Sweep.csv",
        tags: ["write", "unicast", "L1", "multi_core", "atomics"],
        directory: "atomics",
    },
];

const INTERNAL_COLUMNS = new Set([
    "Run Host ID",
    "Architecture",
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

const DEFAULT_X_AXIS = "Transaction Size (bytes)";
const DEFAULT_SERIES = "Number of Transactions";

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

function buildChartConfig(analysis) {
    const { metrics, sweepDimensions } = analysis;

    const bandwidthCol =
        metrics.find((m) => m.includes("Bandwidth")) || null;
    const latencyCol = metrics.find((m) => m.includes("Latency")) || null;

    const xAxisDim = sweepDimensions.find(
        (d) => d.column === DEFAULT_X_AXIS
    );
    const xAxis = xAxisDim ? DEFAULT_X_AXIS : sweepDimensions[0]?.column;

    const groupers = sweepDimensions
        .filter((d) => d.column !== xAxis)
        .map((d) => d.column);

    return { xAxis, groupers, bandwidthCol, latencyCol };
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
    for (const test of TESTS) {
        for (const tag of test.tags) {
            tags.add(tag);
        }
    }
    return [...tags].sort();
}

// ── Filtering ──────────────────────────────────────────────────────

function getFilteredTests() {
    return TESTS.filter((test) => {
        if (
            state.searchQuery &&
            !test.name.toLowerCase().includes(state.searchQuery.toLowerCase())
        ) {
            return false;
        }
        if (state.activeTags.size > 0) {
            for (const tag of state.activeTags) {
                if (!test.tags.includes(tag)) return false;
            }
        }
        return true;
    });
}

// ── Chart Rendering (Plotly) ───────────────────────────────────────

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
        const indices = s.x.map((_, i) => i).sort((a, b) => s.x[a] - s.x[b]);
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

    const layout = {
        xaxis: {
            type: "log",
            dtick: Math.log10(2),
            title: { text: config.xAxis, font: { color: "#a0a0b0" } },
            tickfont: { color: "#a0a0b0" },
            tickformat: "",
            tickvals: getBase2Ticks(traces),
            ticktext: getBase2Ticks(traces).map(formatBytes),
            gridcolor: "#2a2a4a",
            zerolinecolor: "#2a2a4a",
            linecolor: "#2a2a4a",
        },
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
    if (!isFinite(min)) return [];
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
    const filtered = getFilteredTests();

    for (const test of filtered) {
        const li = document.createElement("li");
        if (state.selectedTest && state.selectedTest.name === test.name) {
            li.className = "selected";
        }

        const nameSpan = document.createElement("span");
        nameSpan.textContent = test.name;
        li.appendChild(nameSpan);

        const tagsDiv = document.createElement("div");
        tagsDiv.className = "test-tags";
        for (const tag of test.tags) {
            const tagSpan = document.createElement("span");
            tagSpan.className = "test-tag";
            tagSpan.textContent = tag;
            tagsDiv.appendChild(tagSpan);
        }
        li.appendChild(tagsDiv);

        li.addEventListener("click", () => selectTest(test));
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

function renderSweepFilters(config, rows) {
    const container = document.getElementById("sweep-filters");
    container.innerHTML = "";

    state.visibleValues = new Map();

    for (const dim of config.groupers) {
        const values = [...new Set(rows.map((r) => r[dim]))]
            .sort((a, b) => {
                const na = Number(a), nb = Number(b);
                if (!isNaN(na) && !isNaN(nb)) return na - nb;
                return String(a).localeCompare(String(b));
            });

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

async function selectTest(test) {
    state.selectedTest = test;

    document.getElementById("empty-state").style.display = "none";
    document.getElementById("test-view").style.display = "block";

    document.getElementById("test-name").textContent = test.name;
    document.getElementById("readme-link").href =
        `${GITHUB_BASE}/${test.directory}/README.md`;

    renderTestList();

    state.availableArchs = await probeArchitectures(test);
    if (state.availableArchs.length === 0) {
        document.getElementById("test-info").innerHTML =
            '<span class="info-item"><span class="info-value">No CSV data found for any architecture.</span></span>';
        document.getElementById("arch-chips").innerHTML = "";
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
