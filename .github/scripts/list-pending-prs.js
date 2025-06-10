// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

import {Octokit} from '@octokit/rest';
import {readFileSync, writeFileSync} from 'fs';
import path from 'path';
import {fileURLToPath} from 'url';

// Environment variables
const token       = process.env.GITHUB_TOKEN;
const currentRepo = process.env.GITHUB_REPOSITORY;

if (!token || !currentRepo)
{
    console.error('❌ Missing GITHUB_TOKEN or GITHUB_REPOSITORY.');
    process.exit(1);
}

const [defaultOwner, defaultRepoName] = currentRepo.split('/');
const octokit                         = new Octokit({auth: token});

const getDaysOpen = (createdAt) => Math.floor((new Date() - new Date(createdAt)) / (1000 * 60 * 60 * 24));

const getStyledLabels = (labels) => labels.map((label) => `<span class="label" style="background-color:#${label.color}">${label.name}</span>`).join(' ');

// Read reviewers from reviewers.txt
const __dirname         = path.dirname(fileURLToPath(import.meta.url));
const reviewersFilePath = path.join(__dirname, 'reviewers.txt');
let LLK_TEAM_REVIEWERS;
try
{
    LLK_TEAM_REVIEWERS = new Set(readFileSync(reviewersFilePath, 'utf-8').split('\n').map(line => line.trim()).filter(line => line && !line.startsWith('#')));
}
catch (err)
{
    console.error('❌ Failed to load reviewers.txt:', err);
    process.exit(1);
}

const fetchPullRequests = async ({owner, repo, tag}) => {
    const allPRs = await octokit.paginate(octokit.rest.pulls.list, {
        owner,
        repo,
        state: 'open',
        per_page: 100,
    });

    return allPRs
        .filter(pr => {
            // For tt-metal repo, check if any requested reviewers are from LLK team
            if (owner === 'tenstorrent' && repo === 'tt-metal')
            {
                const reviewers      = pr.requested_reviewers.map(r => r.login);
                const hasLLKReviewer = pr.requested_reviewers.some(reviewer => LLK_TEAM_REVIEWERS.has(reviewer.login));
                return hasLLKReviewer;
            }
            // For tt-llk, include all PRs
            return true;
        })
        .map(pr => ({...pr, _sourceRepo: `${owner}/${repo}`}));
};

const generatePRRow = (pr) => {
    const daysOpen  = getDaysOpen(pr.created_at);
    const createdAt = new Intl.DateTimeFormat('en-US', {day: '2-digit', month: 'long', year: 'numeric'}).format(new Date(pr.created_at));
    const labels    = getStyledLabels(pr.labels);
    const author    = pr.user.login;
    const reviewers = pr.requested_reviewers
                          .map(reviewer => {
                              const login = reviewer.login;
                              return LLK_TEAM_REVIEWERS.has(login) ? `<span class="llk-reviewer">${login}</span>` : login;
                          })
                          .join(', ') ||
                      'None';

    const isDraft = pr.draft;
    const status  = isDraft ? '<span class="status draft">Draft</span>' : '<span class="status ready">Ready</span>';

    // Extract just the repo name from the full repository path
    const repoName = pr._sourceRepo.split('/')[1];

    return `
  <tr class="${isDraft ? 'draft-row' : ''}">
    <td>${repoName}</td>
    <td><a href="${pr.html_url}" target="_blank">#${pr.number}</a></td>
    <td>
      ${status}
      <a href="${pr.html_url}" target="_blank">${pr.title}</a>
    </td>
    <td>${author}</td>
    <td>${reviewers}</td>
    <td>${createdAt}</td>
    <td>${daysOpen}</td>
    <td>${labels}</td>
  </tr>`;
};

const formattedDate = new Intl.DateTimeFormat('en-GB', {day: '2-digit', month: 'long', year: 'numeric'}).format(new Date());

const generateHTMLReport = (rows, allLabels, llkReviewers, allAuthors, authorStats, reviewerStats) => `
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>LLK Pending Pull Requests</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #f4f6f9;
    color: #333;
    padding: 20px;
    line-height: 1.4;
    font-size: 14px;
}
h1 {
    text-align: center;
    color: #444;
    margin-bottom: 20px;
    font-size: 20px;
}
.report-container {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    margin-top: 10px;
    margin-bottom: 10px;
}
.stats-container {
    flex: 2;
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    padding: 15px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.filter-container {
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    padding: 15px;
    margin-bottom: 20px;
    display: flex;
    gap: 20px;
    align-items: center;
}
.filter-group {
    display: flex;
    align-items: center;
    gap: 8px;
}
select {
    padding: 6px 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 14px;
    color: #444;
    background-color: #fff;
}
select:hover {
    border-color: #007bff;
}
select:focus {
    outline: none;
    border-color: #007bff;
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
}
table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 10px;
    background: #fff;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    font-size: 14px;
}
th, td {
    padding: 12px;
    border: 1px solid #ddd;
    text-align: left;
    vertical-align: middle;
}
td {
    font-size: 14px;
}
td a {
    font-size: 14px;
    color: #007bff;
    text-decoration: none;
}
td a:hover {
    text-decoration: underline;
}
th {
    background-color: #007bff;
    color: #fff;
    font-weight: 600;
    white-space: nowrap;
}
th.sortable {
    cursor: pointer;
    position: relative;
    user-select: none;
}
th.sortable:hover {
    background-color: #0056b3;
}
th.sortable::after {
    content: ' ↕';
    opacity: 0.5;
    font-size: 12px;
}
th.sortable.sort-asc::after {
    content: ' ↑';
    opacity: 1;
}
th.sortable.sort-desc::after {
    content: ' ↓';
    opacity: 1;
}
tr:nth-child(even) {
    background-color: #f9f9f9;
}
tr:hover {
    background-color: #f1f5ff;
}
.label {
    display: inline-block;
    padding: 2px 6px;
    margin: 1px 3px;
    font-size: 12px;
    font-weight: 500;
    color: #fff;
    border-radius: 12px;
}
.status {
    display: inline-block;
    padding: 2px 8px;
    margin-right: 8px;
    font-size: 12px;
    font-weight: 500;
    border-radius: 12px;
}
.status.draft {
    background-color: #6c757d;
    color: white;
}
.status.ready {
    background-color: #28a745;
    color: white;
}
.draft-row {
    background-color: #f8f9fa !important;
    color: #6c757d;
}
.draft-row:hover {
    background-color: #e9ecef !important;
}
.draft-row a {
    color: #6c757d;
}
footer {
    text-align: center;
    margin-top: 20px;
    font-size: 12px;
    color: #666;
}
.llk-reviewer {
    font-weight: 600;
    color: #007bff;
    background-color: #e7f0ff;
    padding: 2px 6px;
    border-radius: 4px;
}
.charts-container {
    display: flex;
    gap: 20px;
    margin: 20px 0;
}
.chart-box {
    flex: 1;
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 15px 15px 35px 15px; /* Increased bottom padding */
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    height: 350px; /* Increased height */
    margin-bottom: 20px;
}
.chart-box h3 {
    text-align: center;
    margin-bottom: 15px;
    color: #444;
    font-size: 16px;
}
</style>
</head>
<body>
<h1>LLK Pending Pull Requests — ${formattedDate}</h1>

<div class="charts-container">
    <div class="chart-box">
        <h3>Top 5 PR Authors</h3>
        <canvas id="authorChart"></canvas>
    </div>
    <div class="chart-box">
        <h3>Top 5 LLK Reviewers</h3>
        <canvas id="reviewerChart"></canvas>
    </div>
</div>

<div class="filter-container">
    <div class="filter-group">
        <label for="labelFilter">Filter by Label:</label>
        <select id="labelFilter">
            <option value="">All</option>
            ${allLabels.map(label => `<option value="${label}">${label}</option>`).join('')}
        </select>
    </div>
    <div class="filter-group">
        <label for="llkReviewerFilter">Filter by LLK Reviewer:</label>
        <select id="llkReviewerFilter">
            <option value="">All</option>
            ${llkReviewers.map(reviewer => `<option value="${reviewer}">${reviewer}</option>`).join('')}
        </select>
    </div>
    <div class="filter-group">
        <label for="authorFilter">Filter by Author:</label>
        <select id="authorFilter">
            <option value="">All</option>
            ${allAuthors.map(author => `<option value="${author}">${author}</option>`).join('')}
        </select>
    </div>
    <div class="filter-group">
        <label for="statusFilter">Filter by Status:</label>
        <select id="statusFilter">
            <option value="">All</option>
            <option value="draft">Draft</option>
            <option value="ready">Ready for Review</option>
        </select>
    </div>
    <div class="filter-group">
        <label for="sortOrder">Sort by:</label>
        <select id="sortOrder">
            <option value="days_open">Days Open</option>
            <option value="author">Author</option>
            <option value="repository">Repository</option>
            <option value="title">Title</option>
            <option value="created_at">Created At</option>
        </select>
    </div>
</div>

<table id="prsTable">
<thead>
    <tr>
        <th class="sortable" data-sort="repository">Repository</th>
        <th>PR</th>
        <th class="sortable" data-sort="title">Title</th>
        <th class="sortable" data-sort="author">Author</th>
        <th>Reviewers</th>
        <th class="sortable" data-sort="created_at">Created At</th>
        <th class="sortable" data-sort="days_open">Days Open</th>
        <th>Labels</th>
    </tr>
</thead>
<tbody>
    ${rows}
</tbody>
</table>

<script>
document.addEventListener('DOMContentLoaded', () => {
    // Initialize the charts first
    const authorCtx = document.getElementById('authorChart').getContext('2d');
    const reviewerCtx = document.getElementById('reviewerChart').getContext('2d');

    const authorData = ${JSON.stringify(authorStats)};
    const reviewerData = ${JSON.stringify(reviewerStats)};

    new Chart(authorCtx, {
        type: 'bar',
        data: {
            labels: authorData.map(d => d.name),
            datasets: [{
                label: 'Open PRs',
                data: authorData.map(d => d.count),
                backgroundColor: '#007bff',
                borderColor: '#0056b3',
                borderWidth: 1,
                barPercentage: 0.5,
                categoryPercentage: 0.8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: {
                padding: {
                    left: 5,
                    right: 5,
                    top: 10,
                    bottom: 20 // Increased bottom padding
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                },
                x: {
                    ticks: {
                        padding: 8, // Increased padding
                        font: {
                            size: 11 // Slightly smaller font
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });

    new Chart(reviewerCtx, {
        type: 'bar',
        data: {
            labels: reviewerData.map(d => d.name),
            datasets: [{
                label: 'PRs to review',
                data: reviewerData.map(d => d.count),
                backgroundColor: '#28a745',
                borderColor: '#1e7e34',
                borderWidth: 1,
                barPercentage: 0.5,
                categoryPercentage: 0.8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: {
                padding: {
                    left: 5,
                    right: 5,
                    top: 10,
                    bottom: 20 // Increased bottom padding
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                },
                x: {
                    ticks: {
                        padding: 8, // Increased padding
                        font: {
                            size: 11 // Slightly smaller font
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });

    // Existing filter and sort code
    const labelFilter = document.getElementById('labelFilter');
    const llkReviewerFilter = document.getElementById('llkReviewerFilter');
    const authorFilter = document.getElementById('authorFilter');
    const statusFilter = document.getElementById('statusFilter');
    const sortOrder = document.getElementById('sortOrder');
    const tableBody = document.querySelector('#prsTable tbody');
    const originalRows = Array.from(tableBody.rows);

    // Track current sort state
    let currentSort = 'days_open';
    let currentDirection = 'desc';

    const updateHeaderVisuals = () => {
        // Remove all sort classes
        document.querySelectorAll('th.sortable').forEach(th => {
            th.classList.remove('sort-asc', 'sort-desc');
        });

        // Add current sort class
        const currentHeader = document.querySelector('th[data-sort="' + currentSort + '"]');
        if (currentHeader) {
            currentHeader.classList.add('sort-' + currentDirection);
        }

        // Update dropdown to match
        sortOrder.value = currentSort;
    };

    const filterAndSortTable = () => {
        const label = labelFilter.value;
        const llkReviewer = llkReviewerFilter.value;
        const author = authorFilter.value;
        const status = statusFilter.value;

        let filteredRows = originalRows.filter(row => {
            const rowLabels = Array.from(row.cells[7].querySelectorAll('.label'))
                .map(labelSpan => labelSpan.textContent);
            const rowReviewers = row.cells[4].textContent.split(', ');
            const rowAuthor = row.cells[3].textContent;
            const isDraft = row.classList.contains('draft-row');

            const statusMatch = !status ||
                (status === 'draft' && isDraft) ||
                (status === 'ready' && !isDraft);
            const labelMatch = !label || rowLabels.includes(label);
            const reviewerMatch = !llkReviewer || rowReviewers.includes(llkReviewer);
            const authorMatch = !author || rowAuthor === author;

            return labelMatch && statusMatch && reviewerMatch && authorMatch;
        });

        switch (currentSort) {
            case 'days_open':
                filteredRows.sort((a, b) => {
                    const comparison = parseInt(b.cells[6].textContent) - parseInt(a.cells[6].textContent);
                    return currentDirection === 'asc' ? -comparison : comparison;
                });
                break;
            case 'author':
                filteredRows.sort((a, b) => {
                    const comparison = a.cells[3].textContent.localeCompare(b.cells[3].textContent);
                    return currentDirection === 'desc' ? -comparison : comparison;
                });
                break;
            case 'repository':
                filteredRows.sort((a, b) => {
                    const comparison = a.cells[0].textContent.localeCompare(b.cells[0].textContent);
                    return currentDirection === 'desc' ? -comparison : comparison;
                });
                break;
            case 'title':
                filteredRows.sort((a, b) => {
                    // Extract title from the HTML (skip status span)
                    const titleA = a.cells[2].querySelector('a').textContent;
                    const titleB = b.cells[2].querySelector('a').textContent;
                    const comparison = titleA.localeCompare(titleB);
                    return currentDirection === 'desc' ? -comparison : comparison;
                });
                break;
            case 'created_at':
                filteredRows.sort((a, b) => {
                    const dateA = new Date(a.cells[5].textContent);
                    const dateB = new Date(b.cells[5].textContent);
                    const comparison = dateB.getTime() - dateA.getTime();
                    return currentDirection === 'asc' ? -comparison : comparison;
                });
                break;
        }

        tableBody.innerHTML = '';
        filteredRows.forEach(row => tableBody.appendChild(row));
        updateHeaderVisuals();
    };

    // Header click handlers for sorting
    document.querySelectorAll('th.sortable').forEach(header => {
        header.addEventListener('click', () => {
            const sortType = header.getAttribute('data-sort');

            if (currentSort === sortType) {
                // Toggle direction if clicking the same column
                currentDirection = currentDirection === 'asc' ? 'desc' : 'asc';
            } else {
                // Set new sort column with default direction
                currentSort = sortType;
                currentDirection = sortType === 'days_open' || sortType === 'created_at' ? 'desc' : 'asc';
            }

            filterAndSortTable();
        });
    });

    // Dropdown change handlers
    labelFilter.addEventListener('change', filterAndSortTable);
    llkReviewerFilter.addEventListener('change', filterAndSortTable);
    authorFilter.addEventListener('change', filterAndSortTable);
    statusFilter.addEventListener('change', filterAndSortTable);

    sortOrder.addEventListener('change', () => {
        currentSort = sortOrder.value;
        filterAndSortTable();
    });

    // Initial sort by days open
    updateHeaderVisuals();
    filterAndSortTable();
});
</script>
</body>
</html>
`;

(async () => {
    try
    {
        const localPRs = await fetchPullRequests({owner: defaultOwner, repo: defaultRepoName, tag: 'tt-llk'});

        const ttMetalPRs = await fetchPullRequests({owner: 'tenstorrent', repo: 'tt-metal', tag: 'tt-metal'});

        const relevantTtMetalPRs = ttMetalPRs.filter(pr => {
            const reviewers      = pr.requested_reviewers.map(r => r.login);
            const hasLLKReviewer = reviewers.some(reviewer => LLK_TEAM_REVIEWERS.has(reviewer));
            return hasLLKReviewer;
        });

        const combinedPRs = [...localPRs, ...relevantTtMetalPRs];
        const sortedPRs   = combinedPRs.sort((a, b) => getDaysOpen(b.created_at) - getDaysOpen(a.created_at));

        const allLabels = [...new Set(combinedPRs.flatMap(pr => pr.labels.map(label => label.name)))].sort();

        const llkReviewers =
            [...new Set(combinedPRs.flatMap(pr => pr.requested_reviewers.map(r => r.login).filter(login => LLK_TEAM_REVIEWERS.has(login))))].sort();

        const allAuthors = [...new Set(combinedPRs.map(pr => pr.user.login))].sort();

        // Calculate author statistics
        const authorCounts = {};
        combinedPRs.forEach(pr => {
            const author         = pr.user.login;
            authorCounts[author] = (authorCounts[author] || 0) + 1;
        });

        // Calculate reviewer statistics (only LLK team members)
        const reviewerCounts = {};
        combinedPRs.forEach(pr => {
            pr.requested_reviewers.forEach(reviewer => {
                if (LLK_TEAM_REVIEWERS.has(reviewer.login))
                {
                    reviewerCounts[reviewer.login] = (reviewerCounts[reviewer.login] || 0) + 1;
                }
            });
        });

        // Get top 5 authors
        const authorStats = Object.entries(authorCounts).map(([name, count]) => ({name, count})).sort((a, b) => b.count - a.count).slice(0, 5);

        // Get top 5 reviewers
        const reviewerStats = Object.entries(reviewerCounts).map(([name, count]) => ({name, count})).sort((a, b) => b.count - a.count).slice(0, 5);

        const rows = sortedPRs.map(generatePRRow).join('');
        const html = generateHTMLReport(rows, allLabels, llkReviewers, allAuthors, authorStats, reviewerStats);

        writeFileSync('pending-prs.html', html);
        console.log('✅ HTML report generated: pending-prs.html');
    }
    catch (err)
    {
        console.error('❌ Failed to generate report:', err);
        process.exit(1);
    }
})();
