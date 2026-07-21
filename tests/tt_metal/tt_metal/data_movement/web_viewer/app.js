// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// ── Configuration ──────────────────────────────────────────────────

const FETCH_YAML_FROM_GITHUB = true;
const GITHUB_YAML_BRANCH = "ryanzhu/dm-web";
const GITHUB_RAW_BASE =
    `https://raw.githubusercontent.com/tenstorrent/tt-metal/${GITHUB_YAML_BRANCH}/tests/tt_metal/tt_metal/data_movement`;

const DATA_BASE_PATH = "../data";
const KNOWN_ARCHITECTURES = ["blackhole", "wormhole_b0"];
const LOCAL_GROUPS_YAML_PATH = "../python/test_mappings/web_viewer_groups.yaml";
const LOCAL_TEST_INFO_YAML_PATH = "../python/test_mappings/test_information.yaml";
const GITHUB_BASE =
    "https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/data_movement";

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

let testGroups = [];

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
    groupsYaml: null,
};

// ── YAML Loading ──────────────────────────────────────────────────

async function loadYamlConfig() {
    const groupsPath = FETCH_YAML_FROM_GITHUB
        ? `${GITHUB_RAW_BASE}/python/test_mappings/web_viewer_groups.yaml`
        : LOCAL_GROUPS_YAML_PATH;
    const testInfoPath = FETCH_YAML_FROM_GITHUB
        ? `${GITHUB_RAW_BASE}/python/test_mappings/test_information.yaml`
        : LOCAL_TEST_INFO_YAML_PATH;

    const [groupsResp, testInfoResp] = await Promise.all([
        fetch(groupsPath).catch(() => null),
        fetch(testInfoPath).catch(() => null),
    ]);

    if (groupsResp && groupsResp.ok) {
        state.groupsYaml = jsyaml.load(await groupsResp.text());
    }

    if (testInfoResp && testInfoResp.ok) {
        const testInfo = jsyaml.load(await testInfoResp.text());
        testGroups = buildTestGroups(testInfo.tests, state.groupsYaml || {});
    }
}

function buildTestGroups(tests, groups) {
    const dirMap = new Map();

    for (const [, info] of Object.entries(tests)) {
        if (!info.web_viewer_name || !info.directory) continue;

        if (!dirMap.has(info.directory)) {
            dirMap.set(info.directory, []);
        }
        dirMap.get(info.directory).push({
            name: info.web_viewer_name,
            csv: info.name + ".csv",
        });
    }

    const result = [];
    for (const [directory, testList] of dirMap) {
        const groupEntry = groups[directory];
        result.push({
            group: groupEntry ? groupEntry.name : directory,
            directory,
            tags: groupEntry && groupEntry.tags ? groupEntry.tags : [],
            tests: testList,
        });
    }

    result.sort((a, b) => a.group.localeCompare(b.group));
    return result;
}

function renderFAQ(group) {
    const section = document.getElementById("faq-section");
    const list = document.getElementById("faq-list");
    list.innerHTML = "";

    const yamlEntry = state.groupsYaml && state.groupsYaml[group.directory];
    const faq = yamlEntry && yamlEntry.faq;

    if (!faq || faq.length === 0) {
        section.style.display = "none";
        return;
    }

    section.style.display = "";

    for (const item of faq) {
        const div = document.createElement("div");
        div.className = "faq-item";

        const question = document.createElement("div");
        question.className = "faq-question";

        const chevron = document.createElement("span");
        chevron.className = "faq-chevron";
        chevron.textContent = "▶";

        const qText = document.createElement("span");
        qText.textContent = item.q;

        question.appendChild(chevron);
        question.appendChild(qText);

        const answer = document.createElement("div");
        answer.className = "faq-answer";
        answer.textContent = item.a;

        question.addEventListener("click", () => {
            div.classList.toggle("open");
        });

        div.appendChild(question);
        div.appendChild(answer);
        list.appendChild(div);
    }
}

function renderDiagram(group) {
    const section = document.getElementById("diagram-section");
    const img = document.getElementById("diagram-img");
    const item = document.getElementById("diagram-item");

    const yamlEntry = state.groupsYaml && state.groupsYaml[group.directory];
    const imagePath = yamlEntry && yamlEntry.image;

    if (!imagePath) {
        section.style.display = "none";
        return;
    }

    img.src = `${DATA_BASE_PATH}/${imagePath}`;
    img.alt = group.group + " pattern diagram";
    section.style.display = "";
    item.classList.remove("open");

    const toggle = item.querySelector(".diagram-toggle");
    const newToggle = toggle.cloneNode(true);
    toggle.parentNode.replaceChild(newToggle, toggle);
    newToggle.addEventListener("click", () => {
        item.classList.toggle("open");
    });
}

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
    for (const group of testGroups) {
        for (const tag of group.tags) {
            tags.add(tag);
        }
    }
    return [...tags].sort();
}

// ── Filtering ──────────────────────────────────────────────────────

function getFilteredGroups() {
    const query = state.searchQuery.toLowerCase();
    return testGroups.filter((group) => {
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
    renderDiagram(group);
    renderFAQ(group);

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

async function init() {
    await loadYamlConfig();
    renderTagChips();
    renderTestList();

    document.getElementById("search-input").addEventListener("input", (e) => {
        state.searchQuery = e.target.value;
        renderTestList();
    });
}

document.addEventListener("DOMContentLoaded", init);
